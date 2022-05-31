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


class RandomDummyTrainer(BaseStrategy):
    """
    Perform falcon strategy on the search space.
    """
    def __init__(self, stc_cls, mutated_stc, loss, metrics, meta_metric,
                 resume_dir, device, train_dataset, test_dataset, 
                 lamb=0.01, batch_size=128, workers=4, lr_decay=0.1, eps=1e-4, transfer=False, 
                 max_inner_loop=100, n_hidden=64, n_layers=3, lr=1e-1, log_frequency=5,
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
        self.device = device if device is not None \
                      else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.search_space = {}
        # for name, child in mutated_stc.named_children():
        #     # print(name)
        #     if isinstance(child, nn.LayerChoice):
        #         self.search_space[child._label] = list([str(i) for i in range(len(child.candidates))])
        #     if isinstance(child, nn.InputChoice):
        #         self.search_space[child._label] = list(range(child.n_candidates))

        # very ugly, with rewrite later
        for module in mutated_stc.modules():
            if isinstance(module, nn_.Sequential) or \
               isinstance(module, nn_.ModuleList) :
                for name, child in module.named_modules():
                    if isinstance(child, nn.LayerChoice):
                        self.search_space[child._label] = list([str(i) for i in range(len(child.candidates))])
                    elif isinstance(child, mutables.LayerChoice):
                        self.search_space[name] = list([str(i) for i in range(len(child.names))])
                    elif isinstance(child, ENASMacroLayer):
                        for child_name, subchild in child.named_modules():
                            if isinstance(subchild, nn.LayerChoice):
                                self.search_space[subchild._label] = list([str(i) for i in range(len(subchild.candidates))])
                            elif isinstance(subchild, mutables.LayerChoice):
                                self.search_space[name] = list([str(i) for i in range(len(subchild.names))])
                    if isinstance(child, nn.InputChoice):
                        self.search_space[child._label] = list(range(child.n_candidates))
                    elif isinstance(child, mutables.InputChoice):
                        self.search_space[name] = list(range(child.n_candidates))
            else:
                child = module
                if isinstance(child, nn.LayerChoice):
                        self.search_space[child._label] = list([str(i) for i in range(len(child.candidates))])
                elif isinstance(child, mutables.LayerChoice):
                    self.search_space[name] = list([str(i) for i in range(len(child.names))])
                elif isinstance(child, ENASMacroLayer):
                    for subchild in child.modules():
                        if isinstance(child, nn.LayerChoice) or isinstance(child, mutables.LayerChoice):
                            self.search_space[child._label] = list([str(i) for i in range(len(child.candidates))])
                if isinstance(child, nn.InputChoice):
                    self.search_space[child._label] = list(range(child.n_candidates))
                elif isinstance(child, mutables.InputChoice):
                    self.search_space[name] = list(range(child.n_candidates))
        
        print(self.search_space)
        self.variables = list(self.search_space.keys())
        self.values = list(self.search_space.values())
        self.candidates = grid_list(self.search_space)
        _logger.info('*' * 20 + 'Search Space' + '*' * 20)
        _logger.info(self.search_space)
        _logger.info(f'# Candidates   {len(self.candidates)}')
        print(self.search_space)

        self.sample_size = 1
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.set_prop = set_prop
        self.log_frequency = log_frequency
        self.batch_size = batch_size
        self.workers = workers
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.meta_stc = self.gen_meta_stc(self.values, self.candidates, resume_dir)
        self.meta_stc.to(device)
        self.explored = np.zeros(self.meta_stc.num_nodes, np.bool_)
        _logger.info(self.meta_stc)
        self._init_dataloader()
    
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
        if epoch % self.log_frequency == 0:
            test_meters = AverageMeterGroup()
            for step, data in enumerate(self.test_loader):
                X, y = data
                X, y = to_device(X, self.device), to_device(y, self.device)
                with torch.no_grad():
                    logits, loss = self._logits_and_loss(stc, X, y)
                    metrics = self.metrics(logits, y)
                    test_meters.update(metrics)
            _logger.info('Epoch [%s/%s] %s | test %s', cur_epoch + 1, epoch, meters, test_meters)
    
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


    def get_performance(self, stc, epoch, return_pred=False):
        sample = {key: value for key, value in zip(self.variables, stc)} 
        with ContextStack('fixed', sample):
            stc = self.cls(**self.cls_kwargs)
        stc = stc.to(self.device)
        self.stc_optim = torch.optim.SGD(stc.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=self.weight_decay)
        self.stc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.stc_optim, T_max=200
            )
        for i in range(epoch):
            self._train_one_epoch(i, stc, epoch)
        train_meters, _ = self._eval_stc(stc, 'train')
        val_meters, _ = self._eval_stc(stc, 'val')
        test_meters, _ = self._eval_stc(stc, 'test')
        meters = {'train': train_meters,
                  'val': val_meters,
                  'test': test_meters}
        if return_pred:
            _ , preds = self._eval_stc(stc, 'train')
            return meters, preds
        return meters, None

    def run(self, num_explored=10, visible_epoch=30, max_epoch=200,
            warm_start=False, evaluate=True):
        """
        """
        self.max_epoch = max_epoch
        self.num_explored = num_explored
        self.visible_epoch = visible_epoch
        self.n_fully_train = bound(int(0.1 * num_explored), lower_bound=1, upper_bound=5)

        return_pred = False
        # search
        for self.step in range(num_explored):

            index = self.sample(warm_start=warm_start)
            index = index[0]
            stc = self.candidates[index]
            meters, meta_info = self.get_performance(stc, visible_epoch, return_pred)
            perf = torch.tensor([meters['val']['acc'].avg])

            _logger.info('[step=%d]: Random select %s | val %s | test %s ',
                    self.step, stc,  meters['val'], meters['test'])
            
        if evaluate:
            return self.evaluate()
        else:
            return self.export(explored=True, fully_train=False)

    def sample(self, warm_start=False):
        with torch.no_grad():
            self.n_random_start = bound(
                int(0.1 * self.num_explored), lower_bound=1, upper_bound=10
                )
            indices =  np.random.choice(np.arange(self.meta_stc.num_nodes)[~self.explored], size=1)

        self.last_explored = indices
        self.explored[indices] = True
        return indices


    def export(self, explored=False, fully_train=False):
        if explored:
            pred = self.meta_stc.y
        else:
            raise ValueError
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
