import torch
import torch.nn as nn
import torch.nn.functional as F

from graphgym.config import cfg
from graphgym.models.head_pyg import head_dict
from graphgym.models.layer_pyg import (GeneralLayer, GeneralMultiLayer,
                                       BatchNorm1dNode, BatchNorm1dEdge,
                                       GeneralPoolLayer)
from graphgym.models.act import act_dict
from graphgym.init import init_weights
from graphgym.models.feature_encoder_pyg import node_encoder_dict, \
    edge_encoder_dict

from graphgym.contrib.stage import *
import graphgym.register as register


########### Layer ############
def GNNLayer(dim_in, dim_out, has_act=True):
    return GeneralLayer(cfg.gnn.layer_type, dim_in, dim_out, has_act)


def GNNPreMP(dim_in, dim_out):
    return GeneralMultiLayer('linear', cfg.gnn.layers_pre_mp,
                             dim_in, dim_out, dim_inner=dim_out, final_act=True)


########### Pooling Layer ############ -- [---]
def PoolLayer(dim_in):
    return GeneralPoolLayer(cfg.gnn.pool_type, dim_in)


########### Stage: NN except start and head ############

class GNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module('layer{}'.format(i), layer)
            if i < num_layers - 1 and \
                not cfg.gnn.pool_type=='none' and \
                (i + 2) % cfg.gnn.pool_loop == 0: 
                self.num_layers += 1
                if cfg.gnn.stage_type == 'skipconcat':
                    self.add_module('pool{}'.format(i), PoolLayer(dim_out+d_in))
                else:
                    self.add_module('pool{}'.format(i), PoolLayer(dim_out))
            
    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            # Add pooling layers --[---]
            if isinstance(layer, GeneralPoolLayer):
                batch = layer(batch)
            else:
                x = batch.x
                batch = layer(batch)
                if cfg.gnn.stage_type == 'skipsum':
                    batch.x = x + batch.x
                elif cfg.gnn.stage_type == 'skipconcat' and \
                        i < self.num_layers - 1:
                    batch.x = torch.cat([x, batch.x], dim=1)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch


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

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


########### Model: start + stage + head ############

class GNN(nn.Module):
    '''General GNN model'''

    def __init__(self, dim_in, dim_out, **kwargs):
        super(GNN, self).__init__()
        GNNStage = stage_dict[cfg.gnn.stage_type]
        GNNHead = head_dict[cfg.dataset.task]
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner)
            dim_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNStage(dim_in=dim_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
