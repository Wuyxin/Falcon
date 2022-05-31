import os
import torch
import numpy as np
import logging

import os.path as osp
from graphgym.config import cfg
from graphgym.automl.algorithms.utils.tensor import to_cpu_numpy

logger = logging.getLogger('nas')


class TaskHandler(object):
    
    def __init__(self, task_dict):
        self.train_tasks = task_dict['train_tasks']
        self.test_tasks = task_dict['test_tasks']
        self.n_train = len(self.train_tasks)
        self.n_test = len(self.test_tasks)

        self.stats = {}
        self.train_cnt = np.zeros(self.n_train, dtype=np.int32)
        self.test_cnt = np.zeros(self.n_test, dtype=np.int32)

    def sample(self):
        unseen_flag = self.train_cnt == 0
        if np.sum(unseen_flag):
            task_id = np.random.choice(np.arange(self.n_train)[unseen_flag])
        else:
            task_prob = 1 / self.train_cnt
            task_prob = task_prob / sum(task_prob)
            task_id = np.random.choice(np.arange(self.n_train), 
                                        p=task_prob)
        self.train_cnt[task_id] += 1
        task = self.train_tasks[task_id]
        os.makedirs(osp.join(cfg.out_dir, task), exist_ok=True)
        return task 

    def update_task_records(self, task, records):
        if isinstance(records, torch.Tensor):
            records = to_cpu_numpy(records)
        N = len(records)
        indices = np.arange(N)[records >= 0]
        if not task in self.stats.keys():
            self.stats[task] = {}
        for idx in indices:
            self.stats[task][idx] = records[idx]
        logger.info(f'Stat size of {task} -> {len(self.stats[task])}')

    def task_stat(self, task):
        if task in self.stats.keys():
            records = list(self.stats[task].values())
            records = np.array(records)
            return {'min': np.min(records), 'max': np.max(records),\
                    'mean': np.mean(records), 'std': np.std(records)}
        return None