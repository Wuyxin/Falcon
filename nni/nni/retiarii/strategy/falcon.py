import torch
import logging
import numpy as np
import random
import copy
import itertools
from typing import Any, Dict, List, Sequence, Optional
from nni.nas.pytorch.search_space_zoo import ENASMacroLayer

import torch.nn as nn_
import torch.nn.functional as F
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.nn.dense.linear import Linear

from .base import BaseStrategy
from nni.retiarii.utils import ContextStack
from nni.retiarii.strategy.falcon_module.meta_model import MetaGNN
from nni.retiarii.strategy.falcon_module.meta_stc import gen_meta_graph
from nni.retiarii.strategy.falcon_module.tensor import bound, to_cpu_numpy, to_tensor
from .utils import dry_run_for_search_space, get_targeted_model, filter_model
from nni.retiarii.strategy.falcon_module.subgraph import bid_k_hop_subgraph
from nni.retiarii.experiment.pytorch import preprocess_model
from nni.retiarii.oneshot.pytorch.utils import  AverageMeterGroup, to_device
from nni.retiarii.strategy.falcon_module.label_prop import get_filtered_id
from nni.retiarii.strategy.falcon_module.rank_loss import rank_loss
from nni.retiarii.strategy.falcon_module.phase import phase_space
import nni.retiarii.nn.pytorch as nn
from nni.nas.pytorch import mutables

ROUND = 5
_logger = logging.getLogger('nas')


def grid_list(search_space: Dict[Any, List[Any]], shuffle=True):
    search_space_values = copy.deepcopy(list(search_space.values()))
    if shuffle:
        for values in search_space_values:
            random.shuffle(values)
    candidates = []
    for values in itertools.product(*search_space_values):
        candidates.append(values)
    return candidates


class FalconTrainer(BaseStrategy):
    """
    Perform falcon strategy on the search space.
    """
    def __init__(self, stc_cls, mutated_stc, loss, metrics, meta_metric,
                 resume_dir, device, train_dataset, test_dataset,  
                 lamb=0.01, batch_size=128, workers=4, lr_decay=0.1, eps=1e-4, transfer=False, 
                 max_inner_loop=100, n_hidden=64, n_layers=3, lr=0.1, meta_lr=1e-2, log_frequency=1,
                 weight_decay=5e-4, set_prop=True, cls_kwargs={}, **kwargs):
        self.cls = stc_cls
        self.cls_kwargs = cls_kwargs
        self.lamb = lamb
        self.eps = eps
        self.loss = loss
        self.lr_decay = lr_decay
        self.max_inner_loop = max_inner_loop
        self.metrics = metrics
        self.meta_metric = meta_metric
        self.transfer = transfer
        self.sample_size = 1
        self.lr = lr
        self.meta_lr = meta_lr
        self.weight_decay = weight_decay
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.set_prop = set_prop
        self.log_frequency = log_frequency
        self.batch_size = batch_size
        self.workers = workers
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        self.device = device if device is not None \
                      else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.search_space = phase_space(mutated_stc)
        self.variables = list(self.search_space.keys())
        self.values = list(self.search_space.values())
        self.candidates = grid_list(self.search_space)
        _logger.info('*' * 20 + 'Search Space' + '*' * 20)
        _logger.info(self.search_space)
        _logger.info(f'# Candidates   {len(self.candidates)}')

        self.edge_mask = None
        self.meta_stc = self.gen_meta_stc(self.values, self.candidates, resume_dir)
        self.meta_stc.to(device)
        _logger.info(self.meta_stc)
        self._init_meta_model()
        self._init_dataloader()
    
    def instantialize(self, index):
        choice_dict = {}
        with ContextStack('fixed', choice_dict):
            stc = self.cls(**self.cls_kwargs)
        return stc

    def _init_dataloader(self):
        n_train = len(self.train_dataset)
        split = n_train // 10
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.workers)
        self.val_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=val_sampler,
                                                        num_workers=self.workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                        batch_size=self.batch_size,
                                                        num_workers=self.workers, 
                                                        shuffle=False)
        self.loaders = {'train_loader': self.train_loader, 
                        'val_loader': self.val_loader, 
                        'test_loader':self.test_loader}
        self.train_loader_unshuffled = torch.utils.data.DataLoader(self.train_dataset,
                                                                   batch_size=self.batch_size,
                                                                   num_workers=self.workers,
                                                                   shuffle=False)
        _logger.info(f"#Train: {len(indices[split:])}\n"\
                     f"#Val: {len(indices[:split])}\n"\
                     f"#Test: {len(self.test_loader.dataset)}\n")

    def _train_one_epoch(self, cur_epoch, stc, epoch):
        stc.train()
        meters = AverageMeterGroup()
        for step, data in enumerate(self.train_loader):
            X, y = data
            X, y = to_device(X, self.device), to_device(y, self.device)

            self.stc_optim.zero_grad()
            logits, loss = self._logits_and_loss(stc, X, y)
            loss.backward()
            self.stc_optim.step()
            self.stc_scheduler.step()

            metrics = self.metrics(logits, y)
            metrics['loss'] = loss.item()
            meters.update(metrics)
        return meters

    def _eval_stc(self, stc, split):
        stc.eval()
        meters = AverageMeterGroup()
        preds = torch.tensor([])
        ys = torch.tensor([])
        if split == 'train':
            loader = self.train_loader_unshuffled
        else:
            loader = self.loaders[f'{split}_loader']
        for step, data in enumerate(loader):
            X, y = data
            X = to_device(X, self.device)
            logits = stc(X)
            metrics = self.metrics(logits.detach().cpu(), y.detach().cpu())
            preds = torch.cat([preds, logits.detach().cpu()])
            ys = torch.cat([ys, y])
            meters.update(metrics)
        preds = self.meta_metric(preds, ys)
        return meters, preds

    def _logits_and_loss(self, stc, X, y):
        logits = stc(X)
        loss = self.loss(logits, y)
        return logits, loss

    def gen_meta_stc(self, grid, candidates, resume_dir):
        return gen_meta_graph(grid, candidates, 
                              resume_dir, device=self.device)

    def _init_meta_model(self):
        self.clear_search_status()
        n_node_attr = self.meta_stc.num_node_features 
        n_edge_attr = self.meta_stc.num_edge_features

        self.model = MetaGNN(n_node_attr, n_edge_attr, 
                             self.n_hidden, self.n_layers).to(self.device)
        if self.set_prop:
            self.label_prop = LabelPropagation(num_layers=self.n_layers, 
                                               alpha=0.9).to(self.device)
            self.linear_1 = Linear(-1, self.n_hidden).to(self.device)
            self.linear_2 = torch.nn.Linear(2 * self.n_hidden, 1).to(self.device)

    def _optimizer(self):
        if self.set_prop:
            optimizer = torch.optim.Adam(list(self.model.parameters()) +
                                         list(self.linear_1.parameters()) + 
                                         list(self.linear_2.parameters()), 
                                         lr=self.meta_lr
                                         )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        return optimizer

    
    def get_performance(self, stc, epoch, return_pred=False):
        sample = {key: value for key, value in zip(self.variables, stc)} 
        with ContextStack('fixed', sample):
            stc = self.cls(**self.cls_kwargs)
        stc = stc.to(self.device)
        self.stc_optim = torch.optim.SGD(stc.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=self.weight_decay)
        self.stc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.stc_optim, T_max=epoch
            )
        for i in range(epoch):
            train_meters = self._train_one_epoch(i, stc, epoch)
            if (epoch + 1) % self.log_frequency == 0:
                _logger.info('Epoch [%s/%s] | train %s', i + 1, epoch, train_meters)
        val_meters, _ = self._eval_stc(stc, 'val')
        test_meters, _ = self._eval_stc(stc, 'test')
        meters = {'train': train_meters,
                  'val': val_meters,
                  'test': test_meters}
        if return_pred:
            _ , preds = self._eval_stc(stc, 'train')
            return meters, preds
        return meters, None

    def run(self, num_explored=10, visible_epoch=30, max_epoch=300,
            meta_length=1024, norm=None, zero_shot=False, switch_task=False, 
            warm_start=False, evaluate=True):
        """
        """
        self.max_epoch = max_epoch
        self.num_explored = num_explored
        self.visible_epoch = visible_epoch
        self.n_fully_train = bound(int(0.1 * num_explored), lower_bound=1, upper_bound=5)

        if zero_shot:
            return self.export(explored=False, fully_train=True)
        if switch_task:
            self.clear_search_status()
        return_pred = True
        # search
        for self.step in range(num_explored):

            index = self.sample(warm_start=warm_start)
            index = index[0]
            stc = self.candidates[index]
            meters, meta_info = self.get_performance(stc, visible_epoch, return_pred)
            perf = torch.tensor([meters['val']['acc'].avg])
            self.meta_stc.update_stats(index, perf, meta_info)

            if self.step == self.n_random_start - 1: 
                self.meta_id = get_filtered_id(
                    self.meta_stc.meta_info[self.meta_stc.labeled_nodes],
                    sample_size=meta_length)
                _logger.info(f'Set up meta length as {len(self.meta_id)}')
            if self.step < self.n_random_start:
                _logger.info('[step=%d]: Random select %s | val %s | test %s ',
                    self.step, stc, meters['val'], meters['test'])
            else:
                loss_logger = self.train(norm=norm)
                _logger.info('[step=%d]: %s | val %s | test %s | loss=%f->%f',
                    self.step, stc, 
                    meters['val'], meters['test'],
                    round(loss_logger[0], ROUND), round(loss_logger[-1], ROUND))
            
        if evaluate:
            return self.evaluate()
        else:
            return self.export(explored=True, fully_train=False)

    def clear_search_status(self):
        self.meta_stc.clear_status()
        self.explored = np.zeros(self.meta_stc.num_nodes, np.bool_)
        self.fringe = np.zeros(self.meta_stc.num_nodes, np.bool_)
        self.edge_mask = np.zeros(self.meta_stc.num_edges, np.bool_)

    def pred_score(self):
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

    def sample(self, warm_start=False):
        with torch.no_grad():
            self._eval()
            self.n_random_start = bound(
                int(0.1 * self.num_explored), lower_bound=3, upper_bound=10
                )
            if self.step < self.n_random_start:
                node_perf = None
                indices =  np.random.choice(self.meta_stc.num_nodes, size=1)
            else:
                node_perf = self.pred_score()[self.fringe]
                indices = np.arange(self.meta_stc.num_nodes)[self.fringe]
            
            if node_perf is not None:
                node_perf = to_cpu_numpy(node_perf, keep_dim=False)
                idx = np.argpartition(node_perf, -self.sample_size)[-self.sample_size:]
                indices = indices[idx]

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
               self.lamb * rank_loss(pred, ground_truth)
        return loss

    def train(self, norm=None):
        self._train()
        loss_logger = []
        optimizer = self._optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.max_inner_loop // 10, 
            gamma=self.lr_decay
            )
            
        for _ in range(self.max_inner_loop):
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
        self.linear_1.train()
        self.linear_2.train()
        
    def _eval(self):
        self.model.eval()
        self.linear_1.eval()
        self.linear_2.eval()

    def export(self, explored=False, fully_train=False):
        if explored:
            pred = self.meta_stc.y
        else:
            pred = self.pred_score()  
        index = pred.argmax().item()
        stc = self.candidates[index]
        epoch = self.max_epoch if fully_train else self.visible_epoch
        val_perf, _ = self.get_performance(stc, epoch=epoch)
        test_perf, _ = self.get_performance(stc, epoch=epoch)
        return stc, val_perf, test_perf

    def evaluate(self, split='val'):
        assert split in ['train', 'val']
        '''
        evaluate the algorithm
        :param split: structures are selected based on 
         the performance on valation set, using cfg.metric
        :return: (best structure, performance on valation & test set)
        '''
        self._eval()
        explored_sort = self.meta_stc.y[self.explored].argsort().cpu()
        explored_sort = np.arange(self.meta_stc.num_nodes)[self.explored][explored_sort]
        sel_indices = explored_sort[-self.n_fully_train:]


        fully_trained_stc = [self.candidates[i] for i in sel_indices]
        _logger.info(f"Models to be fully trained: {fully_trained_stc}")
        val_perf = torch.tensor([])
        test_perf = torch.tensor([])
        if len(fully_trained_stc) == 1:
            sel_perf, test_perf = -1, -1
            stc = fully_trained_stc[0]
        else:
            for stc in fully_trained_stc:
                meters, _ = self.get_performance(stc=stc, epoch=self.max_epoch)
                val_perf = torch.cat([val_perf, torch.tensor([meters['val']['acc'].avg])])
                test_perf = torch.cat([test_perf, torch.tensor([meters['test']['acc'].avg])])
            best_idx = val_perf.argmax().item()
            sel_perf, test_perf, stc = val_perf[best_idx], test_perf[best_idx], fully_trained_stc[best_idx]

        _logger.info(f'Best models: {str(stc)}')
        _logger.info(f'Best Validation Performance: {sel_perf}    Test Performance: {test_perf}')
        return stc, sel_perf, test_perf