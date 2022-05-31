import os.path as osp
from graphgym.config import cfg

def set_cfg_task_dir(args, name):
    cfg.dataset.name = name
    try:
        cfg.nas.config_dir = osp.join(args.config_dir, name.lower())
    except:
        cfg.nas.config_dir = osp.join(cfg.nas.config_dir, name.lower())
    try:
        cfg.nas.results_dir = osp.join(args.results_dir, name.lower())
    except:
        cfg.nas.results_dir = osp.join(cfg.nas.results_dir, name.lower())