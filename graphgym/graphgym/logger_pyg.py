import torch
import math
import os
import sys
import logging
import numpy as np
from graphgym.config import cfg
from graphgym.utils.io import dict_to_json, dict_to_tb
import pdb

from sklearn.metrics import *
from tensorboardX import SummaryWriter
from graphgym.utils.device import get_current_gpu_usage


def setup_printing():
    logging.root.handlers = []
    logging_cfg = {'level': logging.INFO, 'format': '%(message)s'}
    h_file = logging.FileHandler('{}/logging.log'.format(cfg.out_dir))
    h_stdout = logging.StreamHandler(sys.stdout)
    if cfg.print == 'file':
        logging_cfg['handlers'] = [h_file]
    elif cfg.print == 'stdout':
        logging_cfg['handlers'] = [h_stdout]
    elif cfg.print == 'both':
        logging_cfg['handlers'] = [h_file, h_stdout]
    else:
        raise ValueError('Print option not supported')
    logging.basicConfig(**logging_cfg)


def hits_K(true, pred_score, K):
    y_pred_pos = pred_score[true==1]
    y_pred_neg = pred_score[true==0]
    
    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    return hitsK


class Logger(object):
    def __init__(self, name='train', task_type=None):
        self.name = name
        self.task_type = task_type
        self.epoch_loss = [] # for performance estimation

        self._epoch_total = cfg.optim.max_epoch
        self._time_total = 0  # won't be reset

        self.out_dir = '{}/{}'.format(cfg.out_dir, name)
        os.makedirs(self.out_dir, exist_ok=True)
        if cfg.tensorboard_each_run:
            self.tb_writer = SummaryWriter(self.out_dir)

        self.reset()

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def reset(self):
        self._iter = 0
        self._size_current = 0
        self._loss = 0
        self._lr = 0
        self._params = 0
        self._time_used = 0
        self._true = []
        self._pred = []
        self._custom_stats = {}

    # basic properties
    def basic(self):
        stats = {'loss': round(self._loss / self._size_current, cfg.round),
                 'lr': round(self._lr, cfg.round),
                 'params': self._params,
                 'time_iter': round(self.time_iter(), cfg.round),
                 }
        gpu_memory = get_current_gpu_usage()
        if gpu_memory > 0:
            stats['gpu_memory'] = gpu_memory
        return stats

    # customized input properties
    def custom(self):
        if len(self._custom_stats) == 0:
            return {}
        out = {}
        for key, val in self._custom_stats.items():
            out[key] = val / self._size_current
        return out

    def _get_pred_int(self, pred_score):
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            return (pred_score > cfg.model.thresh).long()
        else:
            return pred_score.max(dim=1)[1]

    # task properties
    def classification_binary(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        # Error detection --Sihrley
        # AUC is only defined when there is at least one positive data.
        if torch.sum(pred_int == pred_int[0]) == pred_int.numel():
            print("Fail computing auc. Model predictions are all the same!")
            auc = 0
        else:
            auc = round(roc_auc_score(true, pred_int), cfg.round)

        return {'accuracy': round(accuracy_score(true, pred_int), cfg.round),
                'precision': round(precision_score(true, pred_int), cfg.round),
                'recall': round(recall_score(true, pred_int), cfg.round),
                'f1': round(f1_score(true, pred_int), cfg.round),
                'auc': auc,
                }

    def classification_multi(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        return {'accuracy': round(accuracy_score(true, pred_int), cfg.round)}

    def multi_tasks_classification_binary(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        rocauc_list = []
        precision_list = []
        for i in range(true.shape[1]):
            if torch.sum(true[:,i] == 1) > 0 and torch.sum(true[:,i] == 0) > 0:
                is_labeled = true[:,i] == true[:,i]
                # AUC is only defined when there is at least one positive data.
                if torch.sum(true[is_labeled,i] == true[is_labeled,i][0]) == true[is_labeled,i].numel():
                    print("Fail computing auc. Model predictions are all the same!")
                    auc = 0
                else:
                    auc = roc_auc_score(true[is_labeled,i], pred_score[is_labeled,i])
                rocauc_list.append(auc)
                precision_list.append(precision_score(true[is_labeled,i], self._get_pred_int(pred_score[is_labeled,i])))

        return {'auc': round(sum(rocauc_list)/len(rocauc_list), cfg.round),
                'ap': round(sum(precision_list)/len(precision_list), cfg.round)}

    def link_prediction(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        return {'accuracy': round(accuracy_score(true, pred_int), cfg.round),
                'hits_100': round(hits_K(true, pred_score, K=100), cfg.round),
                'hits_50': round(hits_K(true, pred_score, K=50), cfg.round),
                'hits_20': round(hits_K(true, pred_score, K=20), cfg.round)}

    def regression(self):
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        return {'mae': float(round(mean_absolute_error(true, pred), cfg.round)),
                'mse': float(round(mean_squared_error(true, pred), cfg.round)),
                'rmse': float(
                    round(math.sqrt(mean_squared_error(true, pred)), cfg.round))
                }

    def time_iter(self):
        return self._time_used / self._iter

    def eta(self, epoch_current):
        epoch_current += 1  # since counter starts from 0
        time_per_epoch = self._time_total / epoch_current
        return time_per_epoch * (self._epoch_total - epoch_current)

    def update_stats(self, true, pred, loss, lr, time_used, params, **kwargs):
        assert true.shape[0] == pred.shape[0]
        self._iter += 1
        self._true.append(true)
        self._pred.append(pred)
        batch_size = true.shape[0]
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used
        for key, val in kwargs.items():
            if key not in self._custom_stats:
                self._custom_stats[key] = val * batch_size
            else:
                self._custom_stats[key] += val * batch_size

    def write_iter(self):
        raise NotImplementedError

    def write_epoch(self, cur_epoch, estimators=None):
        basic_stats = self.basic()
        custom_stats = self.custom()

        if self.task_type == 'regression':
            task_stats = self.regression()
        elif self.task_type == 'classification_binary':
            if cfg.dataset.task == 'link_pred':
              task_stats = self.link_prediction()
            else:
                task_stats = self.classification_binary()
        elif self.task_type == 'classification_multi':
            task_stats = self.classification_multi()
        elif self.task_type == 'multi_tasks_classification':
            task_stats = self.multi_tasks_classification_binary()
        else:
            raise ValueError('Task has to be regression or classification')

        epoch_stats = {'epoch': cur_epoch}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round)}
        self.epoch_loss.append({**epoch_stats, **basic_stats})

        estimator_stats = {}
        if estimators is not None:
            if not isinstance(estimators, list):
                estimators = [estimators]
            for estimator in estimators:
                estimator_stats[estimator.name] = estimator(self.epoch_loss)

        if self.name == 'train':
            stats = {**epoch_stats, **eta_stats, **basic_stats, **task_stats,
                     **custom_stats, **estimator_stats}
        else:
            stats = {**epoch_stats, **basic_stats, **task_stats, **custom_stats,
                     **estimator_stats}

        # print
        logging.info('{}: {}'.format(self.name, stats))
        # json
        if cur_epoch == 0 and \
            os.path.exists('{}/stats.json'.format(self.out_dir)):
            os.remove('{}/stats.json'.format(self.out_dir))
            
        dict_to_json(stats, '{}/stats.json'.format(self.out_dir))
        # tensorboard
        if cfg.tensorboard_each_run:
            dict_to_tb(stats, self.tb_writer, cur_epoch)
        self.reset()

    def close(self):
        if cfg.tensorboard_each_run:
            self.tb_writer.close()


def infer_task():
    num_label = cfg.share.dim_out
    if cfg.dataset.task_type == 'classification':
        if num_label <= 2:
            task_type = 'classification_binary'
        else:
            task_type = 'classification_multi'
    else:
        task_type = cfg.dataset.task_type
    return task_type


def create_logger():
    loggers = []
    names = ['train', 'val', 'test']
    for i, dataset in enumerate(range(cfg.share.num_splits)):
        loggers.append(Logger(name=names[i], task_type=infer_task()))
    return loggers
