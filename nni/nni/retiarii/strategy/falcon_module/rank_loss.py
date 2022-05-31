# Code adapted from PTRanking

import torch
import torch.nn.functional as F
import numpy as np

def rank_loss(pred: torch.Tensor, ground_truth: torch.Tensor, sigma=0.1):
    # pair-wise rank loss by monte carlo

    size = pred.numel()
    pred = pred.view(1, size)
    pred_dif_ij = torch.unsqueeze(pred, dim=2) - torch.unsqueeze(pred, dim=1)
    pred_prob_ij = torch.sigmoid(pred_dif_ij / sigma)
    
    rank = torch.argsort(ground_truth, descending=False)
    rank = rank.view(1, size)
    dif_ij = torch.unsqueeze(rank, dim=2) - torch.unsqueeze(rank, dim=1)
    prob_ij = torch.clamp(dif_ij, min=-1.0, max=1.0)
    prob_ij = 0.5 * (1.0 + prob_ij)

    _batch_loss = F.binary_cross_entropy(input=torch.triu(pred_prob_ij, diagonal=1),
                                             target=torch.triu(prob_ij, diagonal=1), reduction='none')
    batch_loss = torch.mean(_batch_loss)


    return batch_loss