import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, BatchNorm
from torch_geometric.data import Data 

from functools import wraps


def overload(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) +  len(kwargs) == 2:
            if len(args) == 2: # for inputs like model(g)
                g = args[1]
            else:# for inputs like model(graph=g)
                g = kwargs['graph']
            return func(args[0], g)
        elif len(args) +  len(kwargs) == 4:
            assert len(kwargs) == 3
            return func(args[0], Data(**kwargs))
        else:
            raise TypeError
    return wrapper
        

class MetaGNN(nn.Module):

    def __init__(self, n_node_attr, n_edge_attr, n_hidden, n_layers=2):
        super(MetaGNN, self).__init__()
        
        self.edge_embs = nn.ModuleList([
            nn.Linear(n_edge_attr, n_node_attr) if i == 0 else \
            nn.Linear(n_edge_attr, n_hidden) for i in range(n_layers)
        ])
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                conv = GINEConv(
                    nn=nn.Sequential(nn.Linear(n_node_attr, 2*n_hidden), 
                           nn.ReLU(), 
                           nn.Linear(2*n_hidden, n_hidden))
                )
            else:
                conv = GINEConv(
                    nn=nn.Sequential(nn.Linear(n_hidden, 2*n_hidden), 
                           nn.ReLU(), 
                           nn.Linear(2*n_hidden, n_hidden))
                )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        # MLP is unused for task-specific search or adapt
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, n_hidden*4),
            nn.ReLU(),
            nn.Linear(n_hidden*4, 1),
            )
    
    @overload
    def forward(self, graph):
        x = self.get_node_rep(graph)
        return self.mlp(x)

    @overload
    def get_node_rep(self, graph):
        x = graph.x
        for conv, batch_norm, edge_nn in \
                zip(self.convs, self.batch_norms, self.edge_embs):
            edge_emb = edge_nn(graph.edge_attr)
            x = conv(x, graph.edge_index, edge_emb)
            x = F.relu(batch_norm(x))
        return x