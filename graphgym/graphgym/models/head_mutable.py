""" GNN heads are the last layer of a GNN right before loss computation.

They are constructed in the init function of the gnn.GNN.
"""

import torch
import torch.nn as nn

from graphgym.config import cfg
from graphgym.models.layer_mutable import MLP
from graphgym.models.pooling import pooling_dict

from graphgym.contrib.head import *
import graphgym.register as register


########### Head ############

class GNNNodeHead(nn.Module):
    '''Head of GNN, node prediction'''
    
    # Add kwargs num_layers
    def __init__(self, dim_in, dim_out, num_layers=None):
        super(GNNNodeHead, self).__init__()
        if num_layers is None:
            num_layers = cfg.gnn.layers_post_mp
        self.layer_post_mp = MLP(dim_in, dim_out,
                                 num_layers=num_layers, bias=True)

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        x, edge_index, edge_attr, batch = self.layer_post_mp(x, edge_index, edge_attr, batch)
        return x


class GNNEdgeHead(nn.Module):
    '''Head of GNN, edge prediction'''

    def __init__(self, dim_in, dim_out, num_layers=None):
        ''' Head of Edge and link prediction models.

        Args:
            dim_out: output dimension. For binary prediction, dim_out=1.
        '''
        # Use dim_in for graph conv, since link prediction dim_out could be
        # binary
        # E.g. if decoder='dot', link probability is dot product between
        # node embeddings, of dimension dim_in
        super(GNNEdgeHead, self).__init__()
        # module to decode edges from node embeddings
        
        if num_layers is None:
            num_layers = cfg.gnn.layers_post_mp
        if cfg.model.edge_decoding == 'concat':
            self.layer_post_mp = MLP(dim_in * 2, dim_out,
                                     num_layers=num_layers,
                                     bias=True)
            # requires parameter
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        else:
            if dim_out > 1:
                raise ValueError(
                    'Binary edge decoding ({})is used for multi-class '
                    'edge/link prediction.'.format(cfg.model.edge_decoding))
            self.layer_post_mp = MLP(dim_in, dim_in,
                                     num_layers=num_layers,
                                     bias=True)
            if cfg.model.edge_decoding == 'dot':
                self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
            elif cfg.model.edge_decoding == 'cosine_similarity':
                self.decode_module = nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError('Unknown edge decoding {}.'.format(
                    cfg.model.edge_decoding))

    def _apply_index(self, batch):
        index = '{}_edge_index'.format(batch.split)
        label = '{}_edge_label'.format(batch.split)
        return batch.x[batch[index]], \
               batch[label]

    def forward(self, batch):
        if cfg.model.edge_decoding != 'concat':
            batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        nodes_first = pred[0]
        nodes_second = pred[1]
        pred = self.decode_module(nodes_first, nodes_second)
        return pred, label


class GNNGraphHead(nn.Module):
    '''Head of GNN, graph prediction

    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.
    '''

    def __init__(self, dim_in, dim_out, num_layers=None):
        super(GNNGraphHead, self).__init__()
        if num_layers is None:
            num_layers = cfg.gnn.layers_post_mp
        # todo: PostMP before or after global pooling
        self.layer_post_mp = MLP(dim_in, dim_out,
                                 num_layers=num_layers, bias=True)
        self.pooling_fun = pooling_dict[cfg.model.graph_pooling]

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        graph_emb = self.pooling_fun(x, batch)
        return self.layer_post_mp(graph_emb, edge_index, edge_attr, batch)


# Head models for external interface
head_dict = {
    'node': GNNNodeHead,
    'edge': GNNEdgeHead,
    'link_pred': GNNEdgeHead,
    'graph': GNNGraphHead
}

head_dict = {**register.head_dict, **head_dict}
