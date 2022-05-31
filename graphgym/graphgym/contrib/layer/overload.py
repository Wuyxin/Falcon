
from functools import wraps

def allow_edge_weight(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        assert len(args) +  len(kargs) == 2
        graph = args[1] if len(args) == 2 else kargs['batch']
        if hasattr(graph, 'edge_attr'):
            if graph.edge_attr is not None:
                if graph.edge_attr.dim() == 1 or graph.edge_attr.size(-1) == 1:
                    graph.edge_attr = graph.edge_attr.view(-1,1)
                else:
                    graph.edge_attr = None
        else:
            setattr(graph, 'edge_attr', None)
        return func(args[0], graph)
    return wrapper


def allow_edge_attr(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        assert len(args) +  len(kargs) == 2
        graph = args[1] if len(args) == 2 else kargs['batch']
        if hasattr(graph, 'edge_attr'):
            if graph.edge_attr is not None and graph.edge_attr.dim() == 1:
                graph.edge_attr = graph.edge_attr.view(-1,1)
        else:
            setattr(graph, 'edge_attr', None)
        return func(args[0], graph)
    return wrapper