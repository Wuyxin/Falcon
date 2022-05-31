import os
import os.path as osp
import csv
import yaml
import numpy as np
import pandas as pd
import logging
from texttable import Texttable

from graphgym.config import cfg
from graphgym.utils.io import makedirs_rm_exist, string_to_python
from graphgym.utils.comp_budget import dict_match_baseline
from graphgym.automl.space.rule import RULE_LIST
import graphgym.automl.space as gns

import torch
from itertools import product
from torch.multiprocessing import Manager, Pool, cpu_count

logger = logging.getLogger('nas')
NOPOOL = -100


def get_fname(string):
    if string is not None:
        return string.split('/')[-1].split('.')[0]
    else:
        return 'default'


def load_config(fname):
    if fname is not None:
        with open(fname) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        return {}


def load_search_file(fname):
    with open(fname, 'r') as f:
        out_raw = csv.reader(f, delimiter=' ')
        outs = []
        out = []
        for row in out_raw:
            if '#' in row:
                continue
            elif len(row) > 0:
                assert len(row) == 3, \
                    'Exact 1 space between each grid argument file' \
                    'And no spaces within each argument is allowed'
                out.append(row)
            else:
                if len(out) > 0:
                    outs.append(out)
                out = []
        if len(out) > 0:
            outs.append(out)
    return outs


def grid2list(grid):
    list_in = [[]]
    for grid_temp in grid:
        list_out = []
        for val in grid_temp:
            for list_temp in list_in:
                list_out.append(list_temp + [val])
        list_in = list_out
    return list_in


def to_config(args):
    self, vars = args
    fname_out = self.vars2gnn(vars)
    config_out = self.vars2config(vars)  
    with open('{}/{}.yaml'.format(self.config_dir, fname_out), "w") as f:
        yaml.dump(config_out, f, default_flow_style=False)
    return fname_out


class Grid:

    def __init__(self):
        self.config_base = load_config(cfg.nas.config_base)
        assert self.config_base['optim']["max_epoch"] >= cfg.nas.visible_epoch
        self.config_budget = load_config(cfg.nas.config_budget)
        self.config_grid = cfg.nas.config_grid
        self.fname_start = get_fname(cfg.nas.config_base)

        self.set_space()
        self.update_cfg_dir()

    def update_cfg_dir(self):
        self.config_dir = cfg.nas.config_dir
        self.gen_grid_config()
        self.check()

    def check(self):
        required_attrs = [
            'search_space', # dict, {para: choice, ...}
            '_type',        # list, [type of choice (non-/numeric), ...]
            'vars_list',    # list, [gnn_file_name,...]
            ]
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'Grid is missing {attr_name}.'

    def set_space(self):
        out = load_search_file(self.config_grid)
        while len(out) == 1:
            out = out[0]
        self.vars_label = [row[0].split('.') for row in out]
        self.label = [row[0].split('.')[-1] for row in out]
        self.vars_alias = [row[1] for row in out]
        self.vars_grid = [string_to_python(row[2]) for row in out] 

        if 'pool_type' in self.label and 'pool_loop' in self.label and \
            'none' in self.vars_grid[self.label.index('pool_type')]:
            self.vars_grid[self.label.index('pool_loop')].append(NOPOOL)

        vars_list = grid2list(self.vars_grid)
        self.vars_list = self.remove_duplicate(vars_list)

        self.search_space, self._type = {}, []
        self.space_dict = {} # label: list of candidate values
        for key, label, value in \
            zip(self.vars_alias, self.label, self.vars_grid):
            self.search_space[key] = value
            self.space_dict[label] = value
            self._type.append("numeric" if str(value[0])[0].isnumeric() else 'non-numeric')

    def size(self) -> int:
        return len(self.vars_list)

    def gen_grid_config(self):
        if osp.exists(osp.join(self.config_dir, 'structures.csv')):
            logger.info(f'Using existing configurations in {self.config_dir}')
            return True
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_base['out_dir'] = self.config_dir

        logger.info("Generate configs...It might take a while.")
        gnns_out = []
        pool = Pool(processes = cpu_count() - 1)
        gnns_out = pool.map(to_config, list(product([self], self.vars_list)))

        self.structure_df = pd.DataFrame({"gnn": gnns_out, "variables": self.vars_list})
        self.structure_df.to_csv(f"{self.config_dir}/structures.csv", sep=',')
        logger.info('{} configurations saved to: {}'.format(
            len(self.vars_list), self.config_dir))
    
    def remove_duplicate(self, stc_list):

        logger.info("Solve stucture inconsistency...")
        for id, rule in enumerate(RULE_LIST):
            cnt = 0
            new_stc_list = []
            for idx, stc in enumerate(stc_list):
                if rule(stc, self.label):
                    new_stc_list.append(stc_list[idx])
                else:
                    cnt += 1
            stc_list = new_stc_list
            logger.info(f"Remove {cnt} structures when checking rule {id}.")

        logger.info(f"{len(stc_list)} Left.")
        return stc_list

    def vars2gnn(self, vars):
        gnn_out = self.fname_start
        for id, var in enumerate(vars):
            gnn_out += '-{}={}'.format(self.vars_alias[id],
                                        str(var).strip("[]").strip("''"))
        return gnn_out

    def vars2config(self, vars):
        config_out = self.config_base.copy()
        for id, var in enumerate(vars):
            if len(self.vars_label[id]) == 1:
                config_out[self.vars_label[id][0]] = var
            elif len(self.vars_label[id]) == 2:
                if self.vars_label[id][0] in config_out:  # if key1 exist
                    config_out[self.vars_label[id][0]][self.vars_label[id][1]] = var
                else:
                    config_out[self.vars_label[id][0]] = {self.vars_label[id][1]: var}
            else:
                raise ValueError('Only 2-level config files are supported')
        if len(self.config_budget) > 0:
            config_out = dict_match_baseline(config_out, self.config_budget, verbose=False)
        return config_out

    def index2gnn(self, indices):
        gnn = []
        vars_list = self.index2var(indices)

        for vars in vars_list:
            gnn.append(self.vars2gnn(vars))
        return gnn

    def index2var(self, indices):
        if isinstance(indices, (int, np.int32, np.int64)):
            indices = [indices]
        return np.array(self.vars_list)[indices]

    def choice2gnn(self, choice):
        vars = []
        for id, value in enumerate(self.search_space.values()):
            vars.append(value[choice[id]])
        return self.vars2gnn(vars)

    def __repr__(self) -> str:
        string = f'{self.__class__} Search Space:\n'
        table = Texttable()
        table.add_row(["Parameter", "Value", "Type"])
        for id, (key, value) in enumerate(self.search_space.items()):
            table.add_row([key, value, self._type[id]])
        string += table.draw()
        return string

    def generate_action_list(self, num_of_layers=2):
        action_names = list(self.search_space.keys())
        action_list = action_names * num_of_layers
        return action_list