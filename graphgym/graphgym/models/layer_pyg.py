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
from graphgym.contrib.layer.overload import allow_edge_weight, allow_edge_attr

from graphgym.contrib.layer import *
import graphgym.register as register
from torch_geometric.nn.pool.topk_pool import filter_adj


## General classes
class GeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
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
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class GeneralMultiLayer(nn.Module):
    '''General wrapper for stack of layers'''

    def __init__(self, name, num_layers, dim_in, dim_out, dim_inner=None,
                 final_act=True, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


## Core basic layers
# Input: batch; Output: batch
class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class BatchNorm1dNode(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dEdge, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


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
            layers.append(
                GeneralMultiLayer('linear', num_layers - 1, dim_in, dim_inner,
                                  dim_inner, final_act=True))
            layers.append(Linear(dim_inner, dim_out, bias))
        else:
            layers.append(Linear(dim_in, dim_out, bias))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GCNConv, self).__init__()
        self.model = pyg.nn.GCNConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, batch):
        
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class GraphConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GraphConv, self).__init__()
        self.model = pyg.nn.GraphConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class SAGEConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = pyg.nn.SAGEConv(dim_in, dim_out, **kwargs)
    
    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class TransformerConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(TransformerConv, self).__init__()
        self.model = pyg.nn.TransformerConv(dim_in, dim_out, **kwargs)

    @allow_edge_attr
    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class GATConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GATConv, self).__init__()
        self.model = pyg.nn.GATConv(dim_in, dim_out, **kwargs)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

# Pyg upgrade to 1.8.0
class GATv2Conv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GATv2Conv, self).__init__()
        self.model = pyg.nn.GATv2Conv(dim_in, dim_out, **kwargs)

    @allow_edge_attr
    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class ChebConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(ChebConv, self).__init__()
        self.model = pyg.nn.ChebConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class TAGConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(TAGConv, self).__init__()
        self.model = pyg.nn.TAGConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GINConv, self).__init__()
        gin_nn = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    @allow_edge_weight
    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class ARMAConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(ARMAConv, self).__init__()
        self.model = pyg.nn.ARMAConv(dim_in, dim_out, **kwargs)

    @allow_edge_weight
    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class SplineConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(SplineConv, self).__init__()
        self.model = pyg.nn.SplineConv(dim_in, dim_out,
                                       dim=1, kernel_size=2, **kwargs)

    @allow_edge_attr
    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class GeneralConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GeneralConv, self).__init__()
        self.model = GeneralConvLayer(dim_in, dim_out, **kwargs)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class GeneralEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GeneralEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, **kwargs)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index,
                                edge_feature=batch.edge_attr)
        return batch


class GeneralSampleEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GeneralSampleEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, **kwargs)

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < cfg.gnn.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_attr[edge_mask, :]
        batch.x = self.model(batch.x, edge_index,
                                        edge_feature=edge_feature)
        return batch


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
            
    def forward(self, batch):
        return self.pool_layer(batch)


class SAGPooling(nn.Module):
    def __init__(self, dim_in):
        super(SAGPooling, self).__init__()
        self.model = pyg.nn.SAGPooling(dim_in, ratio=cfg.gnn.pool_ratio)

    def forward(self, batch):
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, _, _ = self.model(batch.x, batch.edge_index,
                            batch.edge_attr, batch.batch)
        return batch


class TopKPooling(nn.Module):
    def __init__(self, dim_in):
        super(TopKPooling, self).__init__()
        self.model = pyg.nn.TopKPooling(dim_in, ratio=cfg.gnn.pool_ratio)

    def forward(self, batch):
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, _, _ = self.model(batch.x, batch.edge_index,
                            batch.edge_attr, batch.batch)
        return batch


class PANPooling(nn.Module):
    def __init__(self, dim_in):
        super(PANPooling, self).__init__()
        self.model = pyg.nn.PANPooling(dim_in, ratio=cfg.gnn.pool_ratio)

    def forward(self, batch):
        N = batch.x.size(0)
        M = SparseTensor.from_edge_index(
                    batch.edge_index,
                    batch.edge_attr.view(-1) if batch.edge_attr.size(-1) == 1\
                        else torch.ones_like(batch.edge_index[0,:])
                    )
        M_ = SparseTensor.from_edge_index(
            batch.edge_index, 
            torch.arange(batch.edge_index.size(-1)).to(batch.x.device)
            )
                    
        row, col, perm_ = M_.coo()
        edge_attr = batch.edge_attr[perm_]
        edge_index = torch.stack([col, row], dim=0)
        batch.x, _, _, batch.batch,  perm, _ = self.model(batch.x, M, batch.batch)
        batch.edge_index, batch.edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=N)
        return batch
                    

class MetaEdgePooling(nn.Module):
    def __init__(self, dim_in):
        super(MetaEdgePooling, self).__init__()
        self.model = EdgePooling(dim_in, dropout=1-cfg.gnn.pool_ratio)

    def forward(self, batch):
        batch.x, batch.edge_index, batch.batch, _ = self.model(batch.x, batch.edge_index, batch.batch)
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            E, V = batch.edge_index.size(1), batch.edge_attr.size(-1)
            batch.edge_attr = torch.ones((E, V)).to(batch.x.device)
        return batch

pool_dict = {
    'sag': SAGPooling,
    'topk': TopKPooling,
    'pan': PANPooling,
    'edge': MetaEdgePooling,
}
