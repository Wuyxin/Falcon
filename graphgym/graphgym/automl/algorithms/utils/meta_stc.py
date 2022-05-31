import os
import os.path as osp
import numpy as np
from copy import copy
from itertools import product

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.multiprocessing import Pool, cpu_count

from graphgym.config import cfg
from graphgym.automl.space.grid import NOPOOL, Grid
from graphgym.automl.space.rule import check_all_rules, query_index


class MetaGraph(Data):

    def __init__(self, x, 
                 edge_index=None, 
                 edge_attr=None, 
                 augment=False, 
                 **kwargs):

        if augment:
            x = torch.cat([x, -torch.ones(x.size(0), 1)], dim=-1)
        super(MetaGraph, self).__init__(x, edge_index, edge_attr, **kwargs)
        self.clear_status()
        self.augment = augment
        self.labeled_nodes = []
        
    def update_stats(self, indices, ys_est, meta_info=None):
        
        if meta_info is not None:
            if meta_info.dim() == 1:
                meta_info = meta_info.view(1, -1)
            assert meta_info.size(0) == len(ys_est)
            self.meta_info = torch.zeros((self.num_nodes, meta_info.size(-1))).to(cfg.device)
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(ys_est, list):
            ys_est = [ys_est]
        for i in range(len(indices)):
            idx, y_est = indices[i], ys_est[i]
            self.y[idx] = y_est.to(cfg.device)
            self.labeled_nodes.append(int(idx))
            if self.augment:
                self.x[idx,-1] = self.y[idx].item()
            if meta_info is not None:
                self.meta_info[idx] = meta_info[i].to(cfg.device)
        self.labeled_nodes.sort()
        self.labeled_nodes = list(set(self.labeled_nodes))
        
    def clear_status(self):
        self.y = -torch.ones(self.num_nodes) * 1e4
        self.y = self.y.to(cfg.device)
        self.meta_info = None


def load_metagraph_files(dir):
    x = torch.load(osp.join(dir , 'x.pt')).float()
    edge_index = torch.load(osp.join(dir , 'edge_index.pt')).long()
    edge_attr = torch.load(osp.join(dir , 'edge_attr.pt')).float()
    return x, edge_index, edge_attr


def nearest_neighbors(args):
    vars, grid, labels = args
    pool_loop_idx = query_index(labels, 'pool_loop')
    pool_type_idx = query_index(labels, 'pool_type')
    check_flag = False
    if pool_loop_idx and pool_type_idx:
        check_flag = True

    neighbors = []
    for k in range(len(vars)):
        # Warning! We assert all valid params are large then zero, 
        # while negative values are used to handle consistency-related problem
        if not isinstance(vars[k], str) and vars[k] < 0:
            continue
        idx_k = grid[k].index(vars[k])
        if isinstance(vars[k], str):
            for l in range(len(grid[k])):
                if l == idx_k:
                    continue
                vars_ = copy(vars)
                vars_[k] =  grid[k][l]
                if check_flag:
                    if vars[k] == 'none':
                        vars_[pool_loop_idx] =  grid[pool_loop_idx][0]
                    elif grid[k][l] == 'none':
                        vars_[pool_loop_idx] = NOPOOL
                if check_all_rules(vars_, labels):
                    neighbors.append(vars_)
                else:
                    raise ValueError
        else:
            if idx_k + 1 < len(grid[k]):
                vars_ = copy(vars)
                vars_[k] =  grid[k][idx_k + 1]
                if check_all_rules(vars_, labels):
                    neighbors.append(vars_)
            if idx_k - 1 >= 0:
                vars_ = copy(vars)
                vars_[k] =  grid[k][idx_k - 1]
                if check_all_rules(vars_, labels):
                    neighbors.append(vars_)

    return neighbors


def node_feat_preprocess(space: Grid):
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    # create node features
    variables = np.array(space.vars_list)
    numeric = np.array(space._type) == 'numeric'
    enc = OneHotEncoder().fit(variables[:,~numeric])
    x_1 = torch.tensor(enc.transform(variables[:,~numeric]).toarray())

    numeric_variables = np.array(variables[:,numeric], dtype=np.float32)
    numeric_variables[numeric_variables < 0] = 0
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(numeric_variables)
    x_2 = torch.tensor(scaler.transform(numeric_variables))

    return torch.cat([x_1, x_2], dim=-1).float()


def gen_meta_graph(space: Grid, augment=False) -> Data:

    resume_dir = cfg.nas.resume_dir
    print(resume_dir)
    if osp.exists(osp.join(resume_dir, 'edge_attr.pt')):
        x, edge_index, edge_attr = load_metagraph_files(resume_dir)
    else:
        os.makedirs(resume_dir, exist_ok=True)
        fname2index = {space.vars2gnn(space.vars_list[i]): i for i in range(space.size())}
        grid = list(space.search_space.values())
        
        print('Processing node features...')
        x = node_feat_preprocess(space)
        
        # create edge index
        print('Processing edge index & features... It might take a while.')
        pool = Pool(processes = cpu_count() // 2)
        neighbors = pool.map(nearest_neighbors, list(product(space.vars_list, [grid], [space.label])))
        edge_index = torch.tensor([[],[]])
        for i, neighbors_i in enumerate(neighbors):
            edge_index_i = torch.tensor([[],[]])
            for neighbor_i in neighbors_i:
                j = fname2index[space.vars2gnn(neighbor_i)]
                edge_index_i = torch.cat((edge_index_i, torch.tensor([[i,j]]).T), dim=-1)
            edge_index = torch.cat((edge_index, edge_index_i), dim=-1)

        # # without using multiprocessing
        #
        # edge_index = torch.tensor([[],[]])
        # for i in tqdm(range(space.size())):
        #     neighbors = nearest_neighbors(space.vars_list[i], grid)
        #     sub_edge_index = torch.tensor([[],[]])
        #     for neighbor in neighbors:
        #         j = fname2index[space.vars2gnn(neighbor)]
        #         sub_edge_index = torch.cat((sub_edge_index, torch.tensor([[i,j]]).T), dim=-1)
        #     edge_index = torch.cat((edge_index, sub_edge_index), dim=-1)

        print('All Done.')
        edge_index, _ = add_self_loops(edge_index.long())
        row, col = edge_index
        edge_attr = (x[col] - x[row]).float()

        torch.save(x, osp.join(resume_dir , 'x.pt'))
        torch.save(edge_index, osp.join(resume_dir , 'edge_index.pt'))
        torch.save(edge_attr, osp.join(resume_dir , 'edge_attr.pt'))

    return MetaGraph(x, edge_index, edge_attr, augment).to(cfg.device)

