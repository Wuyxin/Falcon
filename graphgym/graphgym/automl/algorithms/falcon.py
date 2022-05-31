import numpy as np
import torch
import logging
import torch.nn.functional as F
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.nn.dense.linear import Linear

from graphgym.config import cfg
from graphgym.automl.algorithms.base_meta import MetaBase
from graphgym.automl.algorithms.utils.explain import explain_search_step
from graphgym.automl.algorithms.utils.meta_model import MetaGNN
from graphgym.automl.algorithms.utils.meta_stc import gen_meta_graph
from graphgym.automl.algorithms.utils.subgraph import bid_k_hop_subgraph
from graphgym.automl.algorithms.utils.tensor import to_cpu_numpy, to_tensor, bound
from graphgym.automl.estimate.scratch import get_performance
from graphgym.automl.algorithms.utils.rank_loss import rank_loss
from graphgym.automl.algorithms.utils.label_prop import get_dist, get_filtered_id

logger = logging.getLogger('nas')


class Falcon(MetaBase):
    
    def gen_meta_stc(self, space):
        return gen_meta_graph(space)

    def init_meta_model(self):
        self.clear_search_status()
        n_node_attr = self.meta_stc.num_node_features 
        n_edge_attr = self.meta_stc.num_edge_features

        self.model = MetaGNN(n_node_attr, n_edge_attr, 
                             self.n_hidden, self.n_layers).to(cfg.device)
        if cfg.nas.transfer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.falcon.lr)
        else:
            self.label_prop = LabelPropagation(num_layers=self.n_layers, 
                                               alpha=cfg.falcon.alpha).to(cfg.device)
            self.linear_1 = Linear(-1, self.n_hidden).to(cfg.device)
            self.linear_2 = torch.nn.Linear(2 * self.n_hidden, 1).to(cfg.device)
            self.optimizer = torch.optim.Adam(list(self.model.parameters()) +
                                              list(self.linear_1.parameters()) + 
                                              list(self.linear_2.parameters()), 
                                              lr=cfg.falcon.lr
                                              )

    def search(self, space, norm=None, zero_shot=False,
               switch_task=False, warm_start=False, evaluate=True):
        """
        params:
            space:        search space in graphgym.automl.space
            norm:         for performance normalization
            zero_shot:    zero-shot prediction for pretrained model
            switch_task:  whether to clear previous search status,
                          used when training for multi-tasks
            warm_start:   initialize the initial search point using pretrained model
            evaluate:     whether to return the evaluation results
        """
        self.space = space
        self.zero_shot = zero_shot
        if zero_shot:
            return self.export(explored=False, fully_train=True)

        if switch_task:
            self.clear_search_status()

        # search
        for self.step in range(cfg.nas.num_explored):
            
            index = self.sample(warm_start=warm_start)
            stc = space.index2gnn(index)
            var = space.index2var(index)
            perf = get_performance(stc, split='val', 
                                   epoch=cfg.nas.visible_epoch)
            meta_info = None
            if not cfg.nas.transfer:
                meta_info = get_dist(stc, epoch=cfg.nas.visible_epoch)
            self.meta_stc.update_stats(index, perf, meta_info)

            # for rank_loss, skip the first step 
            if not cfg.nas.transfer and self.step == self.n_random_start - 1: 
                self.meta_id = get_filtered_id(
                    self.meta_stc.meta_info[self.meta_stc.labeled_nodes],
                    sample_size=cfg.falcon.meta_length)
                logger.info(f'Set up meta length as {len(self.meta_id)}')
            if self.step < self.n_random_start:
                logger.info('[step={}]: Random select {} | perf={} '.format(
                    self.step, var, round(perf.item(), cfg.round)))
            else:
                loss_logger = self.train(norm=norm)
                logger.info('[step={}]: {} | perf={} | loss={}->{}'.format(
                    self.step, var, 
                    round(perf.item(), cfg.round), 
                    round(loss_logger[0], cfg.round), 
                    round(loss_logger[-1], cfg.round)))
        
        if evaluate:
            return self.evaluate(space)
        else:
            return self.export(explored=True, fully_train=False)

    def clear_search_status(self):
        self.meta_stc.clear_status()
        self.explored = np.zeros(self.meta_stc.num_nodes, np.bool_)
        self.fringe = np.zeros(self.meta_stc.num_nodes, np.bool_)
        self.edge_mask = np.zeros(self.meta_stc.num_edges, np.bool_)

    def pred_score(self):
        if cfg.nas.transfer or self.zero_shot:
            return self._task_agnostic_score()
        else:
            return self._task_specific_score()

    def _task_specific_score(self):
        stc_embedding = self.model.get_node_rep(
            x=self.meta_stc.x,
            edge_index=self.meta_stc.edge_index[:, self.edge_mask],
            edge_attr=self.meta_stc.edge_attr[self.edge_mask]
            )
        dist_info = self.label_prop(
            y=self.meta_stc.meta_info[:, self.meta_id],
            edge_index=self.meta_stc.edge_index[:, self.edge_mask]
            )
        dist_info = F.relu(self.linear_1(dist_info))
        task_stc_embedding = torch.cat([stc_embedding, dist_info], dim=-1)
        return self.linear_2(task_stc_embedding).view(-1)

    def _task_agnostic_score(self):
        return self.model(
            x=self.meta_stc.x,
            edge_index=self.meta_stc.edge_index[:, self.edge_mask],
            edge_attr=self.meta_stc.edge_attr[self.edge_mask]
            ).view(-1)

    def sample(self, warm_start=False):
        with torch.no_grad():
            self._eval()
            self.n_random_start = bound(
                int(0.1 * cfg.nas.num_explored), lower_bound=3, upper_bound=10
                )
            if warm_start and self.step == 0:
                node_perf = self.model(self.meta_stc)
                indices = np.arange(self.meta_stc.num_nodes)
            elif self.step < self.n_random_start:
                node_perf = None
                indices =  np.random.choice(self.meta_stc.num_nodes, size=1)
            else:
                node_perf = self.pred_score()[self.fringe]
                indices = np.arange(self.meta_stc.num_nodes)[self.fringe]
            
            if node_perf is not None:
                node_perf = to_cpu_numpy(node_perf, keep_dim=False)
                idx = np.argpartition(node_perf, -self.sample_size)[-self.sample_size:]
                indices = indices[idx]
                
        if cfg.falcon.explain:
            self._eval()
            node_idx = to_tensor(indices)
            explain_search_step(cfg.seed, self.step, self.model, 
                                graph=self.meta_stc, 
                                node_idx=node_idx, 
                                num_hops=self.n_layers
                                )

        subset, _, _, edge_mask = bid_k_hop_subgraph(node_idx=to_tensor(indices), 
                                                    num_hops=self.n_layers, 
                                                    edge_index=self.meta_stc.edge_index
                                                    )
        subset, edge_mask = to_cpu_numpy(subset), to_cpu_numpy(edge_mask)

        self.last_explored = indices
        self.explored[indices] = True
        self.fringe[subset] = True
        self.fringe[self.explored] = False
        self.edge_mask[edge_mask] = True
        return indices

    def objective(self, norm=None):
        pred = self.pred_score()
        pred = pred[self.explored].view(-1)
        ground_truth = self.meta_stc.y[self.explored].view(-1)
        if norm is not None: # min-max norm
            ground_truth = (ground_truth - norm['min']) / (norm['max'] - norm['min'] + 1e-6)
        loss = F.mse_loss(pred, ground_truth, reduction="mean") + \
               cfg.falcon.lamb * rank_loss(pred, ground_truth)
        return loss

    def train(self, norm=None):
        self._train()
        loss_logger = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.falcon.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.falcon.max_inner_loop // 3, 
            gamma=cfg.optim.lr_decay
            )
            
        for _ in range(cfg.falcon.max_inner_loop):
            optimizer.zero_grad()
            loss = self.objective(norm)
            loss.backward()
            optimizer.step()
            loss_logger.append(loss.item())
            scheduler.step()
            if loss < self.eps: 
                break
        self.model.eval()
        return loss_logger
        
    def _train(self):
        self.model.train()
        if not cfg.nas.transfer:
            print("*"*50)
            self.linear_1.train()
            self.linear_2.train()
        
    def _eval(self):
        self.model.eval()
        if not cfg.nas.transfer:
            self.linear_1.eval()
            self.linear_2.eval()

    def export(self, explored=False, fully_train=False):
        if explored:
            pred = self.meta_stc.y
        else:
            pred = self.pred_score()  
        index = pred.argmax().item()
        stc = self.space.index2gnn(index)
        epoch = cfg.optim.max_epoch if fully_train else cfg.nas.visible_epoch
        val_perf = get_performance(stc, split='val', epoch=epoch)
        test_perf = get_performance(stc, split='test', epoch=epoch)
        return stc, val_perf, test_perf

    def evaluate(self, space, split='val'):
        assert split in ['train', 'val']
        '''
        evaluate the algorithm
        :param split: structures are selected based on 
         the performance on validation set, using cfg.metric
        :return: (best structure, performance on validation & test set)
        '''
        self._eval()
        explored_sort = self.meta_stc.y[self.explored].argsort().cpu()
        explored_sort = np.arange(self.meta_stc.num_nodes)[self.explored][explored_sort]
        if cfg.nas.metric in ["tse_ema", "tse_e"]:
            sel_indices = explored_sort[:cfg.nas.n_fully_train]
        else:
            sel_indices = explored_sort[-cfg.nas.n_fully_train:]

        fully_trained_stc = space.index2gnn(sel_indices)
        perf = get_performance(stcs=fully_trained_stc, 
                                split=split, epoch=cfg.optim.max_epoch)
        best_idx = perf.argmax().item()
        sel_perf, stc = perf[best_idx], fully_trained_stc[best_idx]
        test_perf = get_performance(stc, split='test', epoch=cfg.optim.max_epoch)

        logger.info(f'Best models: {str(stc)}')
        return stc, sel_perf, test_perf