
from functools import wraps
import torch

def allow_edge_weight(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        assert len(args) == 5
        x, edge_index, edge_attr, batch = args[1:]
        assert isinstance(x, torch.Tensor)
        assert isinstance(edge_index, torch.Tensor)
        
        if edge_attr is not None:
            if edge_attr.dim() == 1 or edge_attr.size(-1) == 1:
                edge_attr = edge_attr.view(-1,1)
        return func(args[0], x, edge_index, edge_attr, batch)
    return wrapper


def allow_edge_attr(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        assert len(args) == 5
        x, edge_index, edge_attr, batch = args[1:]
        if edge_attr is not None:
            if edge_attr is not None and edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1,1)
        return func(args[0], x, edge_index, edge_attr, batch)
    return wrapper