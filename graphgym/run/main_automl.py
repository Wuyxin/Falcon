import os
import os.path as osp
import argparse
import json
import torch

import sys
from graphgym.config import cfg
from graphgym.logger_common import Logger
from graphgym.automl.space.grid import Grid
from graphgym.automl.algorithms import algorithm_dict
from graphgym.utils.device import auto_select_device
from graphgym.utils.seed import set_seed
from graphgym.automl.algorithms.utils.tensor import bound
from graphgym.utils.io import makedirs_rm_exist
from graphgym.utils.comp_budget import params_count

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Architecture Search for GNNs')
    parser.add_argument('--cfg', dest='nas_cfg_file', help='Config file path', required=True, type=str)
    parser.add_argument('--device', nargs='?', type=str, default='none')
    parser.add_argument('--algo', dest='algorithm', required=True, type=str, choices=algorithm_dict.keys())

    parser.add_argument('--root', nargs='?', type=str, default='run/nas/')
    parser.add_argument('--out_dir', nargs='?', type=str, default='run/models/')
    parser.add_argument('--config_dir', nargs='?', type=str, default=None)
    parser.add_argument('--resume_dir', nargs='?', type=str, default=None)
    parser.add_argument('--model_path', nargs='?', type=str, default=None)
    parser.add_argument('--repeat', nargs='?', type=int, default=1)
    parser.add_argument('--set_budget', nargs='?', type=bool, default=True)
    parser.add_argument('--dataset', nargs='?', type=str, default=None)
    parser.add_argument('--zero_shot', nargs='?', type=bool, default=False)
    
    parser.add_argument('--n_hidden', nargs='?', default=64, type=int)
    parser.add_argument('--n_layers', nargs='?', default=3, type=int)
    parser.add_argument('--visible_epoch', nargs='?', default=30, type=int)
    parser.add_argument('--num_explored', nargs='?', default=10, type=int)
    parser.add_argument('opts', help='See graphgym/config.py', nargs=argparse.REMAINDER)
    return parser.parse_args()

nas_dir=None

def merge_cfg(args):   
    if not args.device == 'none': 
        cfg.cuda_visible = args.device
    cfg.nas.visible_epoch = args.visible_epoch
    cfg.nas.num_explored = args.num_explored
    cfg.nas.n_fully_train = bound(int(0.1*args.num_explored), 
                                  lower_bound=1, 
                                  upper_bound=5)
    cfg.merge_from_file(args.nas_cfg_file)
    cfg.merge_from_list(args.opts) 
    if args.set_budget:    
        cfg.darts.epoch = args.visible_epoch * args.num_explored 
        cfg.enas.epoch = args.visible_epoch * args.num_explored


def set_default_cfg_io(args):
    cfg.seed = args.seed
    exp_name = cfg.nas_dest[:-5].split('/')[-1]
    if 'dataset.name' in args.opts:
        dataset_name = args.opts[args.opts.index('dataset.name') + 1]
        exp_name = osp.join(exp_name, dataset_name)

    algorithm = args.algorithm
    if args.model_path is not None:
        pretrain = args.model_path.split('/')[-1][:-3]
        algorithm = f'{algorithm}_pretrain={pretrain}'
    if args.zero_shot:
        assert args.model_path is not None
        algorithm += '_zero_shot'

    nas_dir = osp.join(args.root, exp_name)
    add_info = f'n={args.num_explored}_vis={args.visible_epoch}'
    cfg.nas.config_dir = osp.join(nas_dir, 'configs') if args.config_dir is None else args.config_dir
    cfg.nas.resume_dir = osp.join(nas_dir, 'meta') if args.resume_dir is None else args.resume_dir
    cfg.out_dir = osp.join(args.out_dir, args.algorithm, exp_name, add_info, f'{args.seed + 1}') \
                  if args.algorithm in ['enas', 'darts'] else  osp.join(args.out_dir, exp_name) 
                  

    os.makedirs(nas_dir, exist_ok=True)
    os.makedirs(cfg.nas.config_dir, exist_ok=True)
    os.makedirs(cfg.nas.resume_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_dir = osp.join(nas_dir, 'log', add_info, algorithm)
    params_dir = osp.join(nas_dir, 'params', add_info, algorithm)
    
    if args.seed == 0:
        makedirs_rm_exist(log_dir)
    os.makedirs(params_dir, exist_ok=True)
    # set global logger
    fname = osp.join(log_dir, f'seed={args.seed}.log')
    return params_dir, Logger.get_logger(name='nas', fname=fname), log_dir


if __name__ == '__main__':

    args = parse_args()
    for args.seed in range(args.repeat):
        set_seed(args.seed)
        cfg.nas_dest = args.nas_cfg_file
        # set cfg & logger
        params_dir, logger, log_dir = set_default_cfg_io(args)
        merge_cfg(args)
        # set visible device
        auto_select_device()
        # define space & algorithm
        space = Grid()
        # merge again for consistency
        merge_cfg(args) 
        algorithm = algorithm_dict[args.algorithm](args, space=space)
        logger.info(space)
        # task search 
        warm_start = False
        if args.model_path is not None:
            warm_start = True
            logger.info(f'Load Pretrained Weight from {args.model_path}')
            algorithm.model.load_state_dict(torch.load(args.model_path))
        structure, val_perf, test_perf = algorithm.search(space, 
                                                          zero_shot=args.zero_shot,
                                                          warm_start=warm_start)
        # log output
        logger.info("Best Structure: {}".format(structure))
        logger.info("Validation Performance: {}".format(round(val_perf.item(), cfg.round)))
        logger.info("Test Performance: {}".format(round(test_perf.item(), cfg.round)))
        # save model & results
        if 'meta' in args.algorithm:
            torch.save(algorithm.model.cpu(), osp.join(params_dir, 'model.pt'))
            
        dict = {'seed': args.seed, 'best_structure': structure, 'params': cfg.params,
                'val_perf': round(val_perf.item(), cfg.round), 'test_perf': round(test_perf.item(), cfg.round)}
        with open(osp.join(log_dir, 'best.json'), 'a') as f:
            json.dump(dict, f)
            f.write('\n')
        Logger.deinit_logger()
