from typing import Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class EdgeAttrConstant(BaseTransform):
    r"""Adds a constant value to each edge attr :obj:`edge_attr`.

    Args:
        value (float, optional): The value to add. (default: :obj:`1.0`)
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """
    def __init__(self, value: float = 1.0, cat: bool = True):
        self.value = value
        self.cat = cat

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            c = torch.full((store.num_edges, 1), self.value, dtype=torch.float)

            if hasattr(store, 'edge_attr') and self.cat:
                edge_attr = store.edge_attr.view(-1, 1) if store.edge_attr.dim() == 1 else store.edge_attr
                store.edge_attr = torch.cat([edge_attr, c.to(edge_attr.device, edge_attr.dtype)], dim=-1)
            else:
                store.edge_attr = c

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'
