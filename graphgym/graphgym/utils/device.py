import torch
import subprocess
import numpy as np
import pdb
from graphgym.config import cfg
import logging
import os



def get_gpu_memory_map():
    '''Get the current gpu usage.'''
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory


def get_current_gpu_usage():
    if cfg.gpu_mem and cfg.device != 'cpu' and torch.cuda.is_available():
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-compute-apps=pid,used_memory',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        current_pid = os.getpid()
        used_memory = 0
        for line in result.strip().split('\n'):
            line = line.split(', ')
            if current_pid == int(line[0]):
                used_memory += int(line[1])
        return used_memory
    else:
        return -1


def auto_select_device(memory_max=8000, memory_bias=200, strategy='random'):
    '''Auto select GPU device'''

    if cfg.device != 'cpu' and torch.cuda.is_available():
        if cfg.device == 'auto':
            try:
                memory_raw = get_gpu_memory_map()
            except subprocess.CalledProcessError:
                memory_raw = np.ones(torch.cuda.device_count())
            if len(cfg.cuda_visible):
                if isinstance(cfg.cuda_visible, list):
                    cuda_visible = cfg.cuda_visible
                elif isinstance(cfg.cuda_visible, str):
                    cuda_visible = eval(cfg.cuda_visible)
                else:
                    raise ValueError
            else:
                cuda_visible = [i for i in range(len(memory_raw))]
            invisible_device = torch.ones(len(memory_raw)).bool()
            invisible_device[cuda_visible] = False
            memory_raw[invisible_device] = 1e6
            if strategy == 'greedy' or np.all(memory_raw > memory_max):
                cuda = np.argmin(memory_raw)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info(
                    'Greedy select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))
            elif strategy == 'random':
                memory = 1 / (memory_raw + memory_bias)
                memory[memory_raw > memory_max] = 0
                memory[invisible_device] = 0
                gpu_prob = memory / memory.sum()
                np.random.seed()
                cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
                np.random.seed(cfg.seed)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info('GPU Prob: {}'.format(gpu_prob.round(2)))
                logging.info(
                    'Random select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))

            cfg.device = 'cuda:{}'.format(cuda)

    else:
        cfg.device = 'cpu'