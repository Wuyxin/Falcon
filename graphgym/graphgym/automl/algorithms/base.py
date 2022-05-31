import torch
import logging
import numpy as np
from graphgym.config import cfg
from graphgym.automl.estimate.scratch import get_performance

logger = logging.getLogger('nas')

class Base(torch.nn.Module):
    
    def __init__(self, args, **kwargs):
        super(Base, self).__init__()
        self.sample_size = 1
        for key, value in kwargs.items():
            setattr(self, key, value)

    def search(self, space):

        raise NotImplementedError()

    def evaluate(self, space, split='val'):
        assert split in ['train', 'val']

        # get the performances of explored 
        stc = space.index2gnn(self.explored_stc)
        perf = np.array(
            get_performance(
                stcs=stc, 
                split=split, 
                epoch=cfg.nas.visible_epoch)
            )

        # select cfg.n_fully_trained models according to the cfg.metric on 
        # validation set and fully train them on the training dataset
        sorted = (-perf).argsort()
        if cfg.nas.metric in ["tse_ema", "tse_e"]:
            sel_indices = sorted[:cfg.nas.n_fully_train]
        else:
            sel_indices = sorted[-cfg.nas.n_fully_train:]

        fully_trained_stc = space.index2gnn(sel_indices)
        perf = np.array(get_performance(
            stcs=fully_trained_stc, 
            split=split, epoch=cfg.optim.max_epoch))

        # return the best model with the best {cfg.metric}
        best_idx = perf.argmax()
        sel_perf, stc = perf[best_idx], fully_trained_stc[best_idx]
        test_perf = get_performance(stc, split='test', epoch=cfg.optim.max_epoch)

        return stc, sel_perf, test_perf