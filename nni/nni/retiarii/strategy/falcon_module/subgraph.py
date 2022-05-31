import copy
import networkx as nx
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes


def relabel(x, edge_index):
        
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    return x, edge_index


def path_graphs(graph, node_pair, max_distance=None):
    
    (s, t) = node_pair
    x, edge_index = graph.x, graph.edge_index
    if max_distance is None:
        max_distance = nx.shortest_path_length(to_networkx(graph), s, t)
        
    _, _, _, edge_mask = k_hop_subgraph(node_idx=torch.tensor([t]), 
                                        num_hops=max_distance, 
                                        edge_index=edge_index
                                        )
    data = Data(x=torch.ones(graph.num_nodes),
                    edge_index=edge_index[:, edge_mask])
    nx_graph = to_networkx(data)

    paths = nx.shortest_simple_paths(nx_graph, s, t)
    paths = list(paths)

    path_graphs = []
    for n in range(len(paths)):
        new_edge_index = torch.LongTensor([[],[]])
        for (i, j) in paths[n]:
            new_edge_index = torch.cat([new_edge_index, 
                                        torch.tensor([[i, j]]).T], dim=-1)
        new_x, new_edge_index = relabel(x, new_edge_index)
        path_graphs.append(Data(x=new_x, edge_index=new_edge_index))
    return path_graphs


def filter_subgraph(graph, edge_mask, perturb_ratio=0, relabel=False):
    
    graph = copy.deepcopy(graph)
    graph.edge_index = graph.edge_index[:, edge_mask]
    graph.edge_attr = graph.edge_attr[edge_mask]

    # random perturbation
    if perturb_ratio > 0:
        E = graph.num_edges
        rand_perserve = torch.randperm(E)[:int(E * (1 - perturb_ratio))]
        graph.edge_index = graph.edge_index[:, rand_perserve]
        graph.edge_attr = graph.edge_attr[rand_perserve]

    if relabel:
        graph.x, graph.edge_index = relabel(graph.x, graph.edge_index)
    return graph


def bid_k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None):
    r"""Modified from torch-geometric.utils.subgraph
    
    Computes the bidirectional k-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.

    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        if len(subsets) > 1:
            node_mask[subsets[-2]] = True
        edge_mask1 = torch.index_select(node_mask, 0, row)
        edge_mask2 = torch.index_select(node_mask, 0, col)
        subsets.append(col[edge_mask1])
        subsets.append(row[edge_mask2])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


if __name__ == "__main__":
    # 0 --> 1 ---> 3
    #       |      |
    #       2 <-----    
    edge_index = torch.LongTensor([[0, 1, 2, 3, 1],
                                   [1, 3, 1, 2, 2]])
    graph = Data(x=torch.ones(4), edge_index=edge_index)
    print(list(path_graphs(graph, s=1, t=2, max_distance=3)))
    print(list(path_graphs(graph, s=1, t=2, max_distance=2)))
    print(list(path_graphs(graph, s=0, t=2, max_distance=1)))
    print(list(path_graphs(graph, s=0, t=2)))