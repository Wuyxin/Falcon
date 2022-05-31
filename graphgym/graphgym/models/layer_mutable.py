import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_sparse import SparseTensor

from graphgym.config import cfg
from graphgym.models.act import act_dict
from graphgym.contrib.layer.generalconv import (GeneralConvLayer,
                                                GeneralEdgeConvLayer)
from graphgym.contrib.layer.edge_pool import EdgePooling
from graphgym.contrib.layer.overload_mutable import allow_edge_weight, allow_edge_attr
import graphgym.register as register
from torch_geometric.nn.pool.topk_pool import filter_adj

import nni.retiarii.nn.pytorch as nni

## General classes
class GeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, space_dict, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()

        variables = list(space_dict.keys())
        self.has_l2norm = has_l2norm
        if 'layer_type' in variables:
            layer_choice = []
            for name in space_dict['layer_type']:
                dict = {}
                if name in cfg.gnn.conv_kwargs[0].keys():
                    dict = cfg.gnn.conv_kwargs[0][name]
                layer_choice.append(layer_dict[name](dim_in, dim_out, **dict, **kwargs))
            self.layer = nni.LayerChoice(layer_choice, label='layer_type')
        else:
            dict = {}
            name = cfg.gnn.layer_type
            if name in cfg.gnn.conv_kwargs[0].keys():
                dict = cfg.gnn.conv_kwargs[0][name]
            self.layer = layer_dict[name](dim_in, dim_out, **dict, **kwargs)

        # adding batchnorm choice works bad for some datasets
        self.bn = nni.LayerChoice([nn.BatchNorm1d(dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom),
                                   Identity()], label='batchnorm')
        
        # self.bn = None
        # if has_bn:
        #     self.bn = nn.BatchNorm1d(dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom)
        if 'dropout' in variables:
            self.dropout_layer = nni.LayerChoice([
                nn.Dropout(p=p, inplace=cfg.mem.inplace) \
                    for p in space_dict['dropout']
            ], label='dropout')
        else:
            self.dropout_layer = nn.Dropout(p=cfg.gnn.dropout, 
                                            inplace=cfg.mem.inplace)
        self.act = None
        if has_act:
            if 'act' in variables:
                self.act = nni.LayerChoice([
                    act_dict[act] for act in space_dict['act']
                ], label='act')
            else:
                self.act = act_dict[cfg.gnn.act]
    
    def post_layer(self, x):
        # if self.bn is not None:
        #     x = self.bn(x)
        x = self.bn(x)
        x = self.dropout_layer(x)
        if self.act is not None:
            x = self.act(x)
        if self.has_l2norm:
            x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x, edge_index, edge_attr, batch = self.layer(x, edge_index, edge_attr, batch)
        x = self.post_layer(x)
        return x, edge_index, edge_attr, batch


## General classes
class GeneralLayer_fixed(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer_fixed, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn and cfg.gnn.batchnorm
        dict = {}
        if name==cfg.gnn.layer_type and name in cfg.gnn.conv_kwargs[0].keys():
            dict = cfg.gnn.conv_kwargs[0][name]
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **dict, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])

    def post_layer(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.dropout_layer(x)
        if self.act is not None:
            x = self.act(x)
        if self.has_l2norm:
            x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x, edge_index, edge_attr, batch = self.layer(x, edge_index, edge_attr, batch)
        x = self.post_layer(x)
        return x, edge_index, edge_attr, batch


## Core basic layers
# Input: batch; Output: batch
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x)
        return x, edge_index, edge_attr, batch


class BatchNorm1dNode(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.bn(x)
        return x, edge_index, edge_attr, batch


class BatchNorm1dEdge(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dEdge, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        edge_attr = self.bn(edge_attr)
        return x, edge_index, edge_attr, batch


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None,
                 num_layers=2, **kwargs):
        '''
        Note: MLP works for 0 layers
        '''
        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        if num_layers > 1:
            layers.append(Linear(dim_in, dim_inner, bias))
            for _ in range(num_layers - 2):
                layers.append(Linear(dim_inner, dim_inner, bias))
            layers.append(Linear(dim_inner, dim_out, bias))
        else:
            layers.append(Linear(dim_in, dim_out, bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        for layer in self.layers:
            x, edge_index, edge_attr, batch = layer(x, edge_index, edge_attr, batch)
        return x, edge_index, edge_attr, batch



class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GCNConv, self).__init__()
        self.model = pyg.nn.GCNConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index, edge_attr)
        return x, edge_index, edge_attr, batch


class GraphConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GraphConv, self).__init__()
        self.model = pyg.nn.GraphConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index, edge_attr)
        return x, edge_index, edge_attr, batch


class SAGEConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = pyg.nn.SAGEConv(dim_in, dim_out, **kwargs)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index)
        return x, edge_index, edge_attr, batch


class TransformerConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(TransformerConv, self).__init__()
        self.model = pyg.nn.TransformerConv(dim_in, dim_out, **kwargs)

    @allow_edge_attr
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index, edge_attr)
        return x, edge_index, edge_attr, batch


class GATConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GATConv, self).__init__()
        self.model = pyg.nn.GATConv(dim_in, dim_out, **kwargs)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index)
        return x, edge_index, edge_attr, batch

# Pyg upgrade to 1.8.0
class GATv2Conv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GATv2Conv, self).__init__()
        self.model = pyg.nn.GATv2Conv(dim_in, dim_out, **kwargs)

    @allow_edge_attr
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index, edge_attr)
        return x, edge_index, edge_attr, batch


class ChebConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(ChebConv, self).__init__()
        self.model = pyg.nn.ChebConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index, edge_attr)
        return x, edge_index, edge_attr, batch


class TAGConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(TAGConv, self).__init__()
        self.model = pyg.nn.TAGConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index, edge_attr)
        return x, edge_index, edge_attr, batch


class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GINConv, self).__init__()
        gin_nn = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    @allow_edge_weight
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index)
        return x, edge_index, edge_attr, batch


class ARMAConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(ARMAConv, self).__init__()
        self.model = pyg.nn.ARMAConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index, edge_attr)
        return x, edge_index, edge_attr, batch


class SplineConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(SplineConv, self).__init__()
        self.model = pyg.nn.SplineConv(dim_in, dim_out,
                                       dim=1, kernel_size=2, **kwargs)

    @allow_edge_attr
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index, edge_attr)
        return x, edge_index, edge_attr, batch


class GeneralConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GeneralConv, self).__init__()
        self.model = GeneralConvLayer(dim_in, dim_out, **kwargs)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index)
        return x, edge_index, edge_attr, batch


class GeneralEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GeneralEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, **kwargs)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x = self.model(x, edge_index, edge_feature=edge_attr)
        return x, edge_index, edge_attr, batch


class GeneralSampleEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GeneralSampleEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, **kwargs)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        edge_mask = torch.rand(edge_index.shape[1]) < cfg.gnn.keep_edge
        edge_index = edge_index[:, edge_mask]
        edge_feature = edge_attr[edge_mask, :]
        x = self.model(x, edge_index, edge_feature=edge_feature)
        return x, edge_index, edge_attr, batch


layer_dict = {
    'linear': Linear,
    'mlp': MLP,
    'gcnconv': GCNConv,
    'graphconv': GraphConv,
    'gatconv': GATConv,
    'gatv2conv': GATv2Conv,
    'transformerconv': TransformerConv,
    'ginconv': GINConv,
    'tagconv': TAGConv,
    'sageconv': SAGEConv,
    'armaconv': ARMAConv,
    'chebconv': ChebConv,
    'splineconv': SplineConv,
    'generalconv': GeneralConv,
    'generaledgeconv': GeneralEdgeConv,
    'generalsampleedgeconv': GeneralSampleEdgeConv,
}

# register additional convs
layer_dict = {**register.layer_dict, **layer_dict}



# Pooling 
class GeneralPoolLayer(nn.Module):
    '''General wrapper for pooling layers'''
    
    def __init__(self, name, dim_in):
        super(GeneralPoolLayer, self).__init__()
        self.pool_layer = pool_dict[name](dim_in)
            
    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        return self.pool_layer(x, edge_index, edge_attr, batch)


class SAGPooling(nn.Module):
    def __init__(self, dim_in):
        super(SAGPooling, self).__init__()
        self.model = pyg.nn.SAGPooling(dim_in, ratio=cfg.gnn.pool_ratio)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x, edge_index, edge_attr, batch, _, _ = self.model(x, edge_index,
                            edge_attr, batch)
        return x, edge_index, edge_attr, batch


class TopKPooling(nn.Module):
    def __init__(self, dim_in):
        super(TopKPooling, self).__init__()
        self.model = pyg.nn.TopKPooling(dim_in, ratio=cfg.gnn.pool_ratio)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x, edge_index, edge_attr, batch, _, _ = self.model(x, edge_index,
                            edge_attr, batch)
        return x, edge_index, edge_attr, batch


class PANPooling(nn.Module):
    def __init__(self, dim_in):
        super(PANPooling, self).__init__()
        self.model = pyg.nn.PANPooling(dim_in, ratio=cfg.gnn.pool_ratio)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        N = x.size(0)
        M = SparseTensor.from_edge_index(
                    edge_index,
                    edge_attr.view(-1) if edge_attr.size(-1) == 1\
                        else torch.ones_like(edge_index[0,:])
                    )
        M_ = SparseTensor.from_edge_index(
            edge_index, 
            torch.arange(edge_index.size(-1)).to(x.device)
            )
                    
        row, col, perm_ = M_.coo()
        edge_attr = edge_attr[perm_]
        edge_index = torch.stack([col, row], dim=0)
        x, _, _, batch,  perm, _ = self.model(x, M, batch)
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=N)
        return x, edge_index, edge_attr, batch
                    

class MetaEdgePooling(nn.Module):
    def __init__(self, dim_in):
        super(MetaEdgePooling, self).__init__()
        self.model = EdgePooling(dim_in, dropout=1-cfg.gnn.pool_ratio)

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x, edge_index, batch, _ = self.model(x, edge_index, batch)
        if hasattr(batch, 'edge_attr') and edge_attr is not None:
            E, V = edge_index.size(1), edge_attr.size(-1)
            edge_attr = torch.ones((E, V)).to(x.device)
        return x, edge_index, edge_attr, batch

pool_dict = {
    'sag': SAGPooling,
    'topk': TopKPooling,
    'pan': PANPooling,
    'edge': MetaEdgePooling,
}
