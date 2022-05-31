import re
import os
import json
import torch
import os.path as osp
import subprocess
import yaml

from graphgym.config import cfg


def get_performance(stcs, epoch, split='train'):
    '''
    :param 
        stcs:           structures (str or list)
        epoch:          train each structure until epoch 
        split:          get the performance from split of the dataset

    :return
        performances:   a tensor indicte the performances of the structures
    '''
    if isinstance(stcs, str):
        stcs = [stcs]
    if cfg.nas.transfer:
        out_dir = osp.join(cfg.out_dir, cfg.dataset.name)
    else:
        out_dir = cfg.out_dir

    perfs = []
    lsts = filter(lambda d: osp.isdir(osp.join(out_dir, d)), 
                  os.listdir(out_dir))
    lsts = list(lsts)

    for stc in stcs:
        yaml_path = osp.join(cfg.nas.config_dir, f'{stc}.yaml')

        if epoch == cfg.optim.max_epoch:
            if not osp.exists(osp.join(out_dir, stc, 'agg/')):
                os.system(f'scp {yaml_path} {yaml_path}_backup')
                train(yaml_path, epoch=epoch)
                os.system(f'mv {yaml_path}_backup {yaml_path}')
            else:
                print(f'Reading log from {stc}.')

        else:
            epochs = []
            for lst in lsts:
                _epoch = re.match(stc+r'-epoch=(\d+)', lst)
                if _epoch: 
                    epochs.append(int(_epoch.group(1)))
                    
            if len(epochs) and max(epochs) >= epoch and \
                osp.exists(osp.join(out_dir, f'{stc}-epoch={max(epochs)}/agg')):
                stc = f'{stc}-epoch={max(epochs)}'
                print(f'Reading log from exsisting trained model (epoch={max(epochs)}).')
            else:
                yaml_path_tag = f'{yaml_path[:-5]}-epoch={epoch}.yaml'
                os.system(f'scp {yaml_path} {yaml_path_tag}')
                stc = f'{stc}-epoch={epoch}'
                train(yaml_path=yaml_path_tag, epoch=epoch)
                os.system(f'rm {yaml_path_tag}')

        perf = get_performance_from_log(stc_dir=osp.join(out_dir, stc), 
                                        split=split, 
                                        epoch=epoch
                                        )
        perfs.append(perf)

    return torch.tensor(perfs).view(-1)


def train(yaml_path, epoch=None):
    '''
    :param
        yaml_path: config of the stucture to train
    '''
    print(f'Set visible device {cfg.cuda_visible}')
    n_repeat = 1 if epoch == cfg.nas.visible_epoch else cfg.nas.repeat
    cmd = f'python -m run.{cfg.nas.train_py} --cfg {yaml_path}  --repeat {n_repeat} \
        --epoch {epoch} cuda_visible {cfg.cuda_visible}'
    
    with open(yaml_path) as stream:
        stc_cfg = yaml.safe_load(stream)
    if not ('dataset' in stc_cfg.keys() and \
        'name' in stc_cfg['dataset'].keys()):
        cmd += f' dataset.name {cfg.dataset.name}'

    if cfg.nas.transfer:
        out_dir = osp.join(cfg.out_dir, cfg.dataset.name)
        cmd += f' dataset.name {cfg.dataset.name} out_dir {out_dir}'
    else:
        cmd += f' out_dir {cfg.out_dir}'
    subprocess.check_call(cmd.split())


def get_performance_from_log(stc_dir, split='train', epoch=None):
    '''
    :param 
        stc_dir: directory for recording the stucture training

    :return
        records: performance estimation
    '''
    epoch = cfg.nas.visible_epoch if epoch==None else epoch
    records = []
    if split == 'test' or split == 'val':
        file_name = osp.join(stc_dir, f'agg/{split}/best.json')
        for line in open(file_name, 'r'):
            record = json.loads(line)
            records.append(record[cfg.nas.metric])
            cfg.params = record['params']
            break
    else:
        file_name = osp.join(stc_dir, f'agg/{split}/stats.json')
        stc_records = None
        for line in open(file_name, 'r'):
            record = json.loads(line)
            if record['epoch'] == epoch - 1:
                stc_records = record[cfg.nas.metric]
                records.append(stc_records)
                cfg.params = record['params']
                break

        if stc_records is None:
            if epoch == -1: # last epoch
                records.append(record[cfg.nas.metric])
                cfg.params = record['params']
            else:
                raise ValueError
        
    return records