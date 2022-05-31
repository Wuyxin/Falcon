import torch
import torch.nn as nn
import torch.nn.functional as F

from graphgym.config import cfg
from graphgym.models.head_mutable import head_dict
from graphgym.models.layer_mutable import (BatchNorm1dNode, BatchNorm1dEdge,
                                            GeneralPoolLayer, MLP)
from graphgym.models.layer_mutable import GeneralLayer
from graphgym.models.act import act_dict
from graphgym.init import init_weights
from graphgym.models.feature_encoder_pyg import node_encoder_dict, \
    edge_encoder_dict

from graphgym.contrib.stage import *
import graphgym.register as register

import nni.retiarii.nn.pytorch as nni


########### Layer ############
def GNNLayer(space_dict, dim_in, dim_out, has_act=True):
    return GeneralLayer(space_dict, dim_in, dim_out, has_act)


def GNNPreMP(dim_in, dim_out, num_layers):
    return MLP(dim_in, dim_out, dim_inner=dim_out, 
               num_layers=num_layers, final_act=True)


########### Pooling Layer ############ -- [---]
def PoolLayer(dim_in, pool_type):
    return GeneralPoolLayer(pool_type, dim_in)


########### Mutable GNNs: start + stage + head  ############
class GNN(nni.Module):
    def __init__(self, dim_in, dim_out, space_dict: dict, **kwargs):
        super(GNN, self).__init__()

        variables = list(space_dict.keys())
        GNNStage = stage_dict[cfg.gnn.stage_type]
        GNNHead = head_dict[cfg.dataset.task]
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            if 'layers_pre_mp' in variables:
                self.pre_mp = nni.LayerChoice([
                    GNNPreMP(dim_in, cfg.gnn.dim_inner, num_layers=i) \
                        for i in space_dict['layers_pre_mp']
                    ], label='layers_pre_mp')
            else:
                self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, 
                                       num_layers=cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if cfg.gnn.layers_mp > 0:
            if 'layers_mp' in variables:
                # Only support fixed number of messge passing layers
                self.mp = GNNStage(space_dict=space_dict,
                                   dim_in=dim_in,
                                   dim_out=cfg.gnn.dim_inner,
                                   num_layers=max(space_dict['layers_mp']))
            else:
                self.mp = GNNStage(space_dict=space_dict,
                                   dim_in=dim_in,
                                   dim_out=cfg.gnn.dim_inner,
                                   num_layers=cfg.gnn.layers_mp)
        
        if 'layers_post_mp' in variables:
            self.post_mp = nni.LayerChoice([
                GNNHead(dim_in, dim_out, num_layers=i) \
                    for i in space_dict['layers_post_mp']
                ], label='layers_post_mp')
        else:
            self.post_mp = GNNHead(dim_in, dim_out, num_layers=cfg.gnn.layers_post_mp)
            
        self.apply(init_weights)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        for module in self.children():
            new_batch = module(x, edge_index, edge_attr, batch, **kwargs)
            if isinstance(new_batch, tuple):
                x, edge_index, edge_attr, batch = new_batch
            elif isinstance(x, torch.Tensor):
                x = new_batch
            else:
                x = new_batch
        return x


########### Stage: NN except start and head ############
class GNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers
    
        Note this makes the space different from the original implementation in graphgym
    '''

    def __init__(self, space_dict, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        self.num_layers = num_layers
        layer_list = []
        for i in range(num_layers):
            # we don't use pool and skipconcat for darts
            layer_choice = GNNLayer(space_dict, dim_in, dim_out)
            layer_list.append(layer_choice)

        self.layers = nn.ModuleList(layer_list)
        self.skipconnect = nni.InputChoice(n_candidates=2, label=f'skipconnect')
        self.layers_mp_vars = None
        if 'layers_mp' in list(space_dict.keys()):
            self.layers_mp_vars = space_dict['layers_mp']
            self.selective_layers = nni.InputChoice(
                n_candidates=len(self.layers_mp_vars), 
                label='layers_mp')

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        xs = []
        for i in range(self.num_layers):
            xs.append(x)
            x, edge_index, edge_attr, batch = self.layers[i](x, edge_index, edge_attr, batch)
            x = self.skipconnect([x, x + xs[i]])

        xs.append(x)
        if self.layers_mp_vars is not None:
            x = self.selective_layers([xs[i] for i in self.layers_mp_vars])
        if cfg.gnn.l2norm:
            x = F.normalize(x, p=2, dim=-1)
        return x, edge_index, edge_attr, batch


stage_dict = {
    'stack': GNNStackStage,
    'skipsum': GNNStackStage,
    'skipconcat': GNNStackStage,
}

stage_dict = {**register.stage_dict, **stage_dict}


########### Feature encoder ############

class FeatureEncoder(nn.Module):
    '''Encoding node/edge features'''

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_inner)
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.edge_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.gnn.edge_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        from torch_geometric.data import Data
        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr
        ).to(x.device)
        for module in self.children():
            data = module(data)
        return data.x, data.edge_index, data.edge_attr, batch


