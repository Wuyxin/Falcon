import os
import os.path as osp

import csv
import json
import argparse
import numpy as np
from pandas import DataFrame

from graphgym.config import cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir', dest='exp_dir', help='Results dir path', type=str
    )
    parser.add_argument(
        '--log', dest='log_dir', help='Dir path containing multiple experiments', type=str
    )
    parser.add_argument(
        '--agg', dest='agg_dir', help='aggregated dir path', type=str
    )
    return parser.parse_args()

def eval(exp_dir):
    # algorithms = filter(lambda d: d in _list, 
    #               os.listdir(exp_dir)) 
    columns = ['algorithm', 'num_agg','test_mean', 'test_std', 'val_mean', 'val_std']
    agg_results = DataFrame(columns=columns)      
    for idx, algorithm in enumerate(os.listdir(exp_dir)):
        test_perf, val_perf = [], []
        path = osp.join(osp.join(exp_dir, algorithm), 'best.json')
        path_comp = path.split('/')
        config, algorithm = path_comp[-3], path_comp[-2]
        if args.agg_dir is None:
            args.agg_dir = args.log_dir
        agg_path = osp.join(args.agg_dir, f'{config}.csv')
        if idx == 0 and osp.exists(agg_path):
            os.remove(agg_path)
        if not os.path.exists(path):
            continue
        for line in open(path, 'r'):
            record = json.loads(line)
            try:
                test_perf.append(record['test_perf'])
                val_perf.append(record['val_perf'])
            except:
                test_perf.append(record['test_acc'])
                val_perf.append(record['val_acc'])
        test_perf, val_perf = np.array(test_perf), np.array(val_perf)
        if len(test_perf) == 0:
            return 
        test_mean = round(test_perf.mean(), cfg.round)
        test_std = round(test_perf.std(), cfg.round)
        val_mean = round(val_perf.mean(), cfg.round)
        val_std = round(val_perf.std(), cfg.round)
        values = [algorithm, len(test_perf), test_mean, test_std, val_mean, val_std]
        agg_results = agg_results.append({columns[i]: values[i] for i in range(len(values))}, 
                                          ignore_index=True)

    agg_results = agg_results.sort_values(['test_mean'], ascending=False).reset_index(drop=True)
    agg_results.to_csv(agg_path)


args = parse_args()
if args.log_dir is not None:
    if os.path.exists(args.log_dir):
        for exp_dir in os.listdir(args.log_dir):
            dir = osp.join(args.log_dir, exp_dir)
            if os.path.isdir(dir):
                eval(dir)
elif args.exp_dir is not None:
    eval(args.exp_dir)
    