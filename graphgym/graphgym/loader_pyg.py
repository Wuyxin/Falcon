import networkx as nx
import time
import logging
import pickle
import copy
import os.path as osp

import torch
from torch_geometric.data import DataLoader

from torch_geometric.datasets import *
import torch_geometric.transforms as T
from torch_scatter import scatter

from graphgym.config import cfg
from graphgym.contrib.loader import *
import graphgym.register as register
from graphgym.models.transform import get_link_label, neg_sampling_transform
from graphgym.utils.dataset_utils import InductiveJointDataset, inductive_rand_split
from graphgym.utils.transform_utils import EdgeAttrConstant

# cause locker error
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset

from torch_geometric.utils import to_undirected
#  torch_geometric.data ->  torch_geometric.loader
from torch_geometric.loader.graph_saint import (GraphSAINTNodeSampler,
                                              GraphSAINTEdgeSampler,
                                              GraphSAINTRandomWalkSampler)
from torch_geometric.loader.cluster import ClusterLoader
from torch_geometric.loader import RandomNodeSampler, NeighborSampler
from torch_geometric.utils import negative_sampling


def load_pyg(name, dataset_dir):
    '''
    load pyg format dataset
    :param name: dataset name
    :param dataset_dir: data directory
    :return: a list of networkx/deepsnap graphs
    '''
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        # --[---]
        size = {'Cora': 2708, 'CiteSeer': 3327, 'PubMed': 19717}
        pre_transform = T.Compose([
            T.RandomNodeSplit(
                num_val=int(size[name] * cfg.dataset.split[1]), 
                num_test=int(size[name] * cfg.dataset.split[-1])),
            T.TargetIndegree(),
        ])
        dataset = Planetoid(dataset_dir, name, pre_transform=pre_transform)
    
    elif name[:3] == 'TU_':
        split_names = ['train_graph_index', 'val_graph_index', 
                        'test_graph_index']
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset = TUDataset(dataset_dir, name, transform=T.Constant())
        # by [---]
        elif name[3:8] == 'Tox21':
            train_set = TUDataset(root=osp.join(dataset_dir, name[3:]), name=f'{name[3:]}_training')
            val_set = TUDataset(root=osp.join(dataset_dir, name[3:]), name=f'{name[3:]}_evaluation')
            test_set = TUDataset(root=osp.join(dataset_dir, name[3:]), name=f'{name[3:]}_testing')
            dataset = InductiveJointDataset([train_set, val_set, test_set])
                
        elif name[3:7] == 'NCI1':
            cfg.dataset.edge_dim = 1
            dataset = TUDataset(root=dataset_dir, name=name[3:], transform=EdgeAttrConstant())
        else:
            dataset = TUDataset(dataset_dir, name[3:])

        if not hasattr(dataset, 'splits'):
            dataset = inductive_rand_split(dataset, cfg.dataset.split, seed=123)
        for i, key in enumerate(dataset.splits.keys()):
            id = dataset.splits[key]
            set_dataset_attr(dataset, split_names[i], id, len(id))

    elif name == 'Karate':
        dataset = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset = Coauthor(dataset_dir, name='CS')
        else:
            dataset = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset = Amazon(dataset_dir, name='Computers')
        else:
            dataset = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))

    return dataset


def index2mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def set_dataset_attr(dataset, name, value, size):
    if not hasattr(dataset, 'data') or dataset.data is None:
        setattr(dataset,'data', {})
    dataset._data_list = None
    dataset.data[name] = value
    # error handling --[---]
    if not hasattr(dataset, 'slices') or dataset.slices is None:
        setattr(dataset, 'slices', {})

    dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)


# --[---]
def add_zeros_x(dataset, dim=1):
    if not hasattr(dataset.data, 'x') or dataset.data.x is None:
        setattr(dataset.data, 'x', 
            torch.zeros((dataset.data.num_nodes, dim), dtype=torch.float))
    return dataset


def load_ogb(name, dataset_dir):
    if name[:4] == 'ogbn':
        dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = ['train_mask', 'val_mask', 'test_mask']
        setattr(dataset, 'slices', {key: None for key in split_names})
        for i, key in enumerate(splits.keys()):
            mask = index2mask(splits[key], size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_names[i], mask, len(mask))
        if name == 'ogbn-proteins':
            dataset.data.x = scatter(dataset.data.edge_attr,
                                    dataset.data.edge_index[0],
                                    dim=0,
                                    dim_size=dataset.data.num_nodes,
                                    reduce='mean')
                                    
            dataset.data.node_species = None
            dataset.data.edge_attr = None
        else:
            edge_index = to_undirected(dataset.data.edge_index)
            set_dataset_attr(dataset, 'edge_index', edge_index, edge_index.shape[1])
        
    elif name[:4] == 'ogbg':
        def add_zeros(data, dim=1):
            data.x = torch.zeros((data.num_nodes, dim), dtype=torch.float)
            return data
        if name == 'ogbg-ppa':
            dataset = PygGraphPropPredDataset(name=name, root=dataset_dir, transform=add_zeros)
        else:
            dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = ['train_graph_index', 'val_graph_index',
                       'test_graph_index']
        for i, key in enumerate(splits.keys()):
            id = splits[key]
            set_dataset_attr(dataset, split_names[i], id, len(id))

    elif name[:4] == "ogbl":
        dataset = PygLinkPropPredDataset(name=name, root=dataset_dir)
        dataset = add_zeros_x(dataset)
        splits = dataset.get_edge_split()

        id = splits['train']['edge'].T
        if cfg.dataset.resample_negative:
            set_dataset_attr(dataset, 'train_pos_edge_index', id, id.shape[1])
            # todo: applying transform for negative sampling is very slow
            dataset.transform = neg_sampling_transform
        else:
            # modified --[---]
            id_neg = negative_sampling(edge_index=id,
                                       num_nodes=dataset.data.num_nodes, # num_nodes=dataset.data.num_nodes[0],
                                       num_neg_samples=id.shape[1])
            id_all = torch.cat([id, id_neg], dim=-1)
            label = get_link_label(id, id_neg)
            set_dataset_attr(dataset, 'train_edge_index', id_all,
                             id_all.shape[1])
            set_dataset_attr(dataset, 'train_edge_label', label, len(label))

        id, id_neg = splits['valid']['edge'].T, splits['valid']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = get_link_label(id, id_neg)
        set_dataset_attr(dataset, 'val_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'val_edge_label', label, len(label))

        id, id_neg = splits['test']['edge'].T, splits['test']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = get_link_label(id, id_neg)
        set_dataset_attr(dataset, 'test_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'test_edge_label', label, len(label))

    else:
        raise ValueError('OGB dataset: {} non-exist')
    return dataset


def load_dataset():
    '''
    load raw datasets.
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    '''
    format = cfg.dataset.format
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        dataset = func(format, name, dataset_dir)
        if dataset is not None:
            return dataset
    # Load from Pytorch Geometric dataset
    if format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    # Load from OGB formatted data
    elif format == 'OGB':
        dataset = load_ogb(name.replace('_', '-'), dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(format))
    return dataset


def set_dataset_info(dataset):
    # set shared variables
    # todo: verify edge cases
    
    # get dim_in and dim_out
    try:
        cfg.share.dim_in = dataset.data.x.shape[1]
    except:
        cfg.share.dim_in = 1
    try:
        cfg.dataset.edge_dim = dataset.data.edge_attr.shape[1]
    except:
        pass
    try: # fix bug --[---]
        if cfg.dataset.task_type == 'classification' and \
            (len(dataset.data.y.shape) == 1 or dataset.data.y.shape[1] == 1):
                cfg.share.dim_out = torch.unique(dataset.data.y).shape[0]
        else:
            cfg.share.dim_out = dataset.data.y.shape[1]
    except:
        cfg.share.dim_out = 1
    # count number of dataset splits
    cfg.share.num_splits = 1
    for key in dataset.data.keys:
        if 'val' in key:
            cfg.share.num_splits += 1
            break
    for key in dataset.data.keys:
        if 'test' in key:
            cfg.share.num_splits += 1
            break


def create_dataset():
    ## todo: add new PyG dataset split functionality
    ## Load dataset
    dataset = load_dataset()

    set_dataset_info(dataset)

    return dataset


def get_loader(dataset, sampler, batch_size, shuffle=True):
    if sampler == "full_batch" or len(dataset) > 1:
        loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "neighbor":
        loader_train = NeighborSampler(dataset[0],
                                       sizes=cfg.train.neighbor_sizes[
                                             :cfg.gnn.layers_mp],
                                       batch_size=batch_size, shuffle=shuffle,
                                       num_workers=cfg.num_workers,
                                       pin_memory=True)
    elif sampler == "random_node":
        loader_train = RandomNodeSampler(dataset[0],
                                         num_parts=cfg.train.train_parts,
                                         shuffle=shuffle,
                                         num_workers=cfg.num_workers,
                                         pin_memory=True)
    elif sampler == "saint_rw":
        loader_train = GraphSAINTRandomWalkSampler(dataset[0],
                                                   batch_size=batch_size,
                                                   walk_length=cfg.train.walk_length,
                                                   num_steps=cfg.train.iter_per_epoch,
                                                   sample_coverage=0,
                                                   shuffle=shuffle,
                                                   num_workers=cfg.num_workers,
                                                   pin_memory=True)
    elif sampler == "saint_node":
        loader_train = GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
                                             num_steps=cfg.train.iter_per_epoch,
                                             sample_coverage=0, shuffle=shuffle,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True)
    elif sampler == "saint_edge":
        loader_train = GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
                                             num_steps=cfg.train.iter_per_epoch,
                                             sample_coverage=0, shuffle=shuffle,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True)
    elif sampler == "cluster":
        loader_train = ClusterLoader(dataset[0],
                                     num_parts=cfg.train.train_parts,
                                     save_dir="{}/{}".format(cfg.dataset.dir,
                                                             cfg.dataset.name.replace(
                                                                 "-", "_")),
                                     batch_size=batch_size, shuffle=shuffle,
                                     num_workers=cfg.num_workers,
                                     pin_memory=True)

    else:
        raise NotImplementedError("%s sampler is not implemented!" % sampler)
    return loader_train


def create_loader(shuffle_train=True):
    dataset = create_dataset()
    print(dataset)
    # train loader
    if cfg.dataset.task == 'graph':
        try:
            id = dataset.data['train_graph_index']
        except:
            id = dataset.data.train_graph_index
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=shuffle_train)]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                              shuffle=shuffle_train)]

    # val and test loaders
    split_names = ['val_graph_index', 'test_graph_index']
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':
            try:
                id = dataset.data[split_names[i]]
            except:
                id = getattr(dataset.data, split_names[i])
            loaders.append(
                get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                get_loader(dataset, cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))

    return loaders
