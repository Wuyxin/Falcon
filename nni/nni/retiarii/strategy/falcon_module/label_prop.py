import os
import yaml
import subprocess
import logging
import os.path as osp

import torch
from torch.distributions import Categorical
from scipy.stats import entropy

logger = logging.getLogger('nas')


# TODO: get dist
def get_dist(stc, epoch):
    '''
    :param 
        stcs:           structures (str or list)
        epoch:          train each structure until epoch 
        split:          get the performance from split of the dataset

    :return
        performances:   a tensor indicte the performances of the structures
    '''        
    if isinstance(stc, list):
        assert len(stc) == 1
        stc = stc[0]
    yaml_path = osp.join(cfg.nas.config_dir, f'{stc}.yaml')
    n_repeat = 1 if epoch == cfg.nas.visible_epoch else cfg.nas.repeat
    cmd = f'python -m run.main_infer --cfg {yaml_path}  --repeat {n_repeat} \
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
        out_dir = cfg.out_dir
        cmd += f' out_dir {out_dir}'

    save_path = os.path.join(out_dir, stc, 'train_dist.pt')
    if not os.path.exists(save_path):
        subprocess.check_call(cmd.split())
    else:
        logger.info(f'Reading performance distribution (epoch={epoch})')
    return torch.load(save_path)


def get_filtered_id(X: torch.tensor, sample_size: int):
    unique = X.unique()
    stat = torch.zeros((len(unique), X.size(-1)))
    for idx, value in enumerate(unique):
        stat[idx] = torch.sum(X == value, dim=0)
    stat = stat / stat.sum(dim=0)
    logits = torch.tensor(entropy(stat, axis=0))
    m = Categorical(logits=logits.view(-1))
    id = m.sample([sample_size])
    return torch.unique(id)