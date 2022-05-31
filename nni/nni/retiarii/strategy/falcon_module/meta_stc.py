import os
import os.path as osp
import numpy as np
from copy import copy
from itertools import product
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.multiprocessing import Pool, cpu_count


class MetaGraph(Data):

    def __init__(self, x, 
                 edge_index=None, 
                 edge_attr=None, 
                 device=None,
                 **kwargs):

        super(MetaGraph, self).__init__(x, edge_index, edge_attr, **kwargs)
        self.device = device
        self.clear_status()
        self.labeled_nodes = []
        
    def update_stats(self, indices, ys_est, meta_info=None):
        
        if meta_info is not None:
            if meta_info.dim() == 1:
                meta_info = meta_info.view(1, -1)
            assert meta_info.size(0) == len(ys_est)
            if not hasattr(self, 'meta_info') or self.meta_info is None:
                self.meta_info = torch.zeros((self.num_nodes, meta_info.size(-1))).to(self.device)
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(ys_est, list):
            ys_est = [ys_est]
        for i in range(len(indices)):
            idx, y_est = indices[i], ys_est[i]
            self.y[idx] = y_est.to(self.device)
            self.labeled_nodes.append(int(idx))
            if meta_info is not None:
                self.meta_info[idx] = meta_info[i].to(self.device)
        self.labeled_nodes.sort()
        self.labeled_nodes = list(set(self.labeled_nodes))
        
    def clear_status(self):
        self.y = -torch.ones(self.num_nodes) * 1e4
        self.y = self.y.to(self.device)
        self.meta_info = None


def load_metagraph_files(resume_dir):
    x = torch.load(osp.join(resume_dir , 'x.pt')).float()
    edge_index = torch.load(osp.join(resume_dir , 'edge_index.pt')).long()
    edge_attr = torch.load(osp.join(resume_dir , 'edge_attr.pt')).float()
    return x, edge_index, edge_attr


def nearest_neighbors(args):
    # vars: list of structures
    # grid: space in grid
    vars, grid = args
    vars = list(vars)
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
                neighbors.append(tuple(vars_))
        else:
            if idx_k + 1 < len(grid[k]):
                vars_ = copy(vars)
                vars_[k] =  grid[k][idx_k + 1]
                neighbors.append(tuple(vars_))
            if idx_k - 1 >= 0:
                vars_ = copy(vars)
                vars_[k] =  grid[k][idx_k - 1]
                neighbors.append(tuple(vars_))

    return neighbors


def node_feat_preprocess(space_list):
    # create node features
    variables = np.array(space_list)
    enc = OneHotEncoder().fit(variables)
    x = torch.tensor(enc.transform(variables).toarray())
    return x


# def node_feat_preprocess(space):
#     # create node features
#     variables = np.array(space.vars_list)
#     numeric = np.array(space._type) == 'numeric'
#     enc = OneHotEncoder().fit(variables[:,~numeric])
#     x_1 = torch.tensor(enc.transform(variables[:,~numeric]).toarray())

#     numeric_variables = np.array(variables[:,numeric], dtype=np.float32)
#     numeric_variables[numeric_variables < 0] = 0
#     scaler = MinMaxScaler(feature_range=(0, 1)).fit(numeric_variables)
#     x_2 = torch.tensor(scaler.transform(numeric_variables))
#     return torch.cat([x_1, x_2], dim=-1).float()


def gen_meta_graph(grid, candidates, resume_dir, device) -> Data:

    
    if osp.exists(osp.join(resume_dir, 'edge_attr.pt')):
        x, edge_index, edge_attr = load_metagraph_files(resume_dir)
    else:
        size = len(candidates)
        os.makedirs(resume_dir, exist_ok=True)
        fname2index = {candidates[i]: i for i in range(size)}
        print('Processing node features...')
        x = node_feat_preprocess(candidates)
        
        # create edge index
        print('Processing edge index & features...It might take a while.')
        pool = Pool(processes = cpu_count() - 1)
        neighbors = pool.map(nearest_neighbors, list(product(candidates, [grid])))
        edge_index = torch.tensor([[],[]])
        for i, neighbors_i in enumerate(neighbors):
            edge_index_i = torch.tensor([[],[]])
            for neighbor_i in neighbors_i:
                j = fname2index[neighbor_i]
                edge_index_i = torch.cat((edge_index_i, torch.tensor([[i,j]]).T), dim=-1)
            edge_index = torch.cat((edge_index, edge_index_i), dim=-1)

        print('All Done.')
        edge_index, _ = add_self_loops(edge_index.long())
        row, col = edge_index
        edge_attr = (x[col] - x[row]).float()
        torch.save(x, osp.join(resume_dir , 'x.pt'))
        torch.save(edge_index, osp.join(resume_dir , 'edge_index.pt'))
        torch.save(edge_attr, osp.join(resume_dir , 'edge_attr.pt'))

    return MetaGraph(x, edge_index, edge_attr , device=device)

