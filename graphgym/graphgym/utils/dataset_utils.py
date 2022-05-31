import numpy as np
from typing import Optional, Union
from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import InMemoryDataset


import copy
from collections.abc import Mapping
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.data.separate import separate



IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class InductiveJointDataset(InMemoryDataset):
    def __init__(self, datasets: list, transform=None, pre_transform=None):

        self.data = None
        self.transform = transform
        self.pre_transform = pre_transform
        self._indices: Optional[Sequence] = None

        max_dim = 0
        for dataset in datasets:
            if dataset[0].x.size(-1) > max_dim:
                max_dim = dataset[0].x.size(-1)

        data_list = []
        for dataset in datasets:
            for i in range(len(dataset)):
                data = dataset[i]
                if data.x.size(-1) < max_dim:
                    data.x = torch.cat(
                        [data.x, data.x.new_zeros(data.x.size(0), max_dim - data.x.size(-1))], 
                        dim=-1)
                data_list.append(data)

        self.data, self.slices = self.collate(data_list)

        train_mask = torch.zeros(len(data_list), dtype=torch.bool)
        val_mask = torch.zeros(len(data_list), dtype=torch.bool)
        test_mask = torch.zeros(len(data_list), dtype=torch.bool)

        indices = torch.arange(len(data_list))
        train_mask[:len(datasets[0])] = True
        val_mask[len(datasets[0]):len(datasets[0]) + len(datasets[1])] = True
        test_mask[-len(datasets[2]):] = True

        dataset.splits = {
            'train': indices[train_mask], 
            'val': indices[val_mask], 
            'test': indices[test_mask]
            }
            

def inductive_rand_split(dataset, split, seed):

    np.random.seed(seed)
    length = len(dataset)
    indices = torch.from_numpy(np.random.permutation(length)).long()
    cum_indices = [int(sum(split[0:x:1]) * length) for x in range(0, len(split) + 1)]

    setattr(
        dataset, 'splits', 
        {'train': indices[:cum_indices[1]],
        'val': indices[cum_indices[1]:cum_indices[2]],
        'test': indices[cum_indices[2]:]}
        )

    return dataset
