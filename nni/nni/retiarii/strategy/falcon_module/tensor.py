import os
import numpy as np

import torch

eps = 1e-4


def min_max_scale(x: torch.tensor, scale=1.0):
    _max, _min = x.max(), x.min()
    return torch.where(
        _max - _min > eps,
        (x - _min)/(_max - _min) * scale,
        (x - _min)/(_max - _min + eps) * scale)
        

def to_cpu_numpy(x: torch.tensor, keep_dim=True):
    x = x.detach().cpu()
    return x.numpy() if keep_dim else x.view(-1).numpy()


def to_tensor(x, device=None):
    if device:
        return torch.tensor(x).to(device)
    return torch.tensor(x)
    

def bound(number, lower_bound=1, upper_bound=5):
    return min(upper_bound, max(number, lower_bound))


def tensor_topk_idx(input, topk, descend=True):
    input = to_cpu_numpy(input)
    if descend:
        return np.argpartition(input, -topk)[-topk:]
    else:
        return np.argpartition(input, topk)[:topk]


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def remove_file(path):
    if os.path.exists(path):
        os.remove(path)