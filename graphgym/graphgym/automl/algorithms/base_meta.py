import torch
import logging
import numpy as np
import torch.nn.functional as F

from graphgym.config import cfg
from graphgym.automl.algorithms.base import Base
from graphgym.automl.estimate.scratch import get_performance
from graphgym.automl.algorithms.utils.tensor import to_cpu_numpy, to_tensor, bound

logger = logging.getLogger('nas')

class MetaBase(Base):
    
    eps = 1e-4
    sample_size = 1

    def __init__(self, args, space, **kwargs):
        super(MetaBase, self).__init__(args, **kwargs)
        assert args.n_hidden is not None
        assert args.n_layers is not None

        self.n_hidden = args.n_hidden
        self.n_layers = args.n_layers
        self.meta_stc = None
        self.edge_mask = None
        self.space = space

        if self.meta_stc is None:
            self.meta_stc = self.gen_meta_stc(space)
        self.init_meta_model()
        logger.info(self.meta_stc)
        
    def search(self, space, norm=None, zero_shot=False,
               clear_status=False, warm_start=False):
        """
        params:
            space:        search space in graphgym.automl.space
            norm:         for performance normalization
            zero_shot:    zero-shot prediction for pretrained model
            clear_status: whether to clear previous search status,
                          used when trained for multi-tasks
            warm_start:   initialize the initial search point using pretrained model
        """
        if zero_shot:
            return self.evaluate(space, zero_shot=True)

        if clear_status:
            self.clear_status()

        # search
        for self.step in range(cfg.nas.num_explored):

            index = self.sample(warm_start=warm_start)
            stc = space.index2gnn(index)
            perf = get_performance(stc, split='val', 
                                   epoch=cfg.nas.visible_epoch
                                   )
            self.meta_stc.update_stats(index, perf)
            # for rank_loss, skip the first step 
            if self.step > 0:
                loss_logger = self.train(norm=norm)

                logger.info('[step={}]: {} | perf={} | loss={}->{}'.format(
                    self.step, np.array(space.vars_list)[index].reshape(-1), 
                    round(perf.item(), cfg.round), 
                    round(loss_logger[0], cfg.round), 
                    round(loss_logger[-1], cfg.round)))
            
        return self.evaluate(space, zero_shot=False)
        
    def evaluate(self, space, split='val', zero_shot=False):
        assert split in ['train', 'val']
        '''
        evaluate the algorithm
        :param split: structures are selected based on 
         the performance on validation set, using cfg.metric
        :return: (best structure, performance on validation & test set)
        '''
        self.model.eval()
        if zero_shot:
            sorted = self.model(self.meta_stc).view(-1).argsort().cpu()
            sorted = np.arange(self.meta_stc.num_nodes)[sorted]
        else:
            sorted = self.meta_stc.y[self.explored].argsort().cpu()
            sorted = np.arange(self.meta_stc.num_nodes)[self.explored][sorted]
        if cfg.nas.metric in ["tse_ema", "tse_e"]:
            sel_indices = sorted[:cfg.nas.n_fully_train]
        else:
            sel_indices = sorted[-cfg.nas.n_fully_train:]

        fully_trained_stc = space.index2gnn(sel_indices)
        perf = get_performance(stcs=fully_trained_stc, 
                                split=split, epoch=cfg.optim.max_epoch)
        perf = np.array(perf)

        best_idx = perf.argmax()
        sel_perf, stc = perf[best_idx], fully_trained_stc[best_idx]
        test_perf = get_performance(stc, split='test', epoch=cfg.optim.max_epoch)

        logger.info(f'Best models: {str(stc)}')
        return stc, sel_perf, test_perf
    
    
    def train(self):
        '''
        training the meta-model
        '''
        raise NotImplementedError()


    def init_meta_model(self):
        '''
        initialize the meta-model if applicable
        '''
        raise NotImplementedError()

    def sample(self, warm_start=False):
        '''
        initialize the meta-model if applicable
        '''
        raise NotImplementedError()

    def gen_meta_stc(self, space):
        '''
        generate meta-structure
        '''
        raise NotImplementedError()


class HrtMetaBase(Base):
    '''
    Require self-designed functions:
        gen_meta_stc:         generate meta-structure
    '''
    def __init__(self, args, **kwargs):
        super(HrtMetaBase, self).__init__(args, **kwargs)

    def search(self, space, **kwargs):
        self.meta_stc = self.gen_meta_stc(space)
        logger.info(self.meta_stc)
        self.explored = np.zeros(self.meta_stc.num_nodes, np.bool_)
        # search
        for self.step in range(cfg.nas.num_explored):
            index = self.sample()
            stc = space.index2gnn(index)
            perf = get_performance(stc, split='val', epoch=cfg.nas.visible_epoch)
            self.meta_stc.update_stats(index, perf)
            self.update_status()
            logger.info('[Step={}]: {} | Perf={} '.format(
                self.step, np.array(space.vars_list)[index].reshape(-1),
                round(perf.item(), cfg.round)))
        return self.evaluate(space)
        
    def evaluate(self, space, split='val'):
        assert split in ['train', 'val']
        '''
        evaluate the algorithm
        :param split: structures are selected based on 
         the performance on {split} set, using metric {cfg.metric}
        :return: (best structure, performance on {split} & test set)
        '''
    
        sorted = self.meta_stc.y[self.explored].argsort().cpu()
        sorted = np.arange(self.meta_stc.num_nodes)[self.explored][sorted]
        if cfg.nas.metric in ["tse_ema", "tse_e"]:
            sel_indices = sorted[:cfg.nas.n_fully_train]
        else:
            sel_indices = sorted[-cfg.nas.n_fully_train:]

        fully_trained_stc = space.index2gnn(sel_indices)
        perf = get_performance(stcs=fully_trained_stc, split=split, epoch=cfg.optim.max_epoch)
        perf = np.array(perf)

        best_idx = perf.argmax()
        sel_perf, stc = perf[best_idx], fully_trained_stc[best_idx]
        test_perf = get_performance(stc, split='test', epoch=cfg.optim.max_epoch)

        return stc, sel_perf, test_perf
    
    def sample(self, warm_start=False):
        '''
        Sample one index from space
        '''
        raise NotImplementedError()

    def gen_meta_stc(self, space):
        '''
        generate meta-structure
        '''
        raise NotImplementedError()

    def update_status(self, space):
        '''
        update search status
        '''
        raise NotImplementedError()