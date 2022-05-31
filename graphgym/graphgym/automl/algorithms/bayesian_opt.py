import logging
import warnings
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from graphgym.config import cfg
from graphgym.automl.algorithms.base import Base
from graphgym.automl.algorithms.utils.tensor import bound
from graphgym.automl.estimate.scratch import get_performance
from graphgym.automl.algorithms.utils.meta_stc import node_feat_preprocess

logger = logging.getLogger('nas')

class BayesianOptimization(Base):

    def __init__(
        self,  
        args,
        acq='poi',
        kappa=2.576,
        kappa_decay=0.9,
        **kwargs
    ):
        super(BayesianOptimization, self).__init__(args, **kwargs)
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=cfg.seed,
        )
        self.acq = acq
        self._kappa = kappa
        self._kappa_decay = kappa_decay

    def get_utility(self, x):
        if self.acq == 'ucb':
            return self._ucb(x)
        if self.acq == 'ei':
            return self._ei(x)
        if self.acq == 'poi':
            return self._poi(x)

    def _ucb(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self.gp.predict(x, return_std=True)
        return mean + self._kappa * std

    def _ei(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self.gp.predict(x, return_std=True)
        a = (mean - self.y_max)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    def _poi(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self.gp.predict(x, return_std=True)
        z = (mean - self.y_max) / std
        return norm.cdf(z)

    def update_params(self):
        if self._kappa_decay < 1:
            self._kappa = bound(self._kappa * self._kappa_decay, 
                                lower_bound=0.1
                                )
        
    def search(self, space, **kwargs):
        x = node_feat_preprocess(space)

        n_init_nodes = bound(int(0.1 * cfg.nas.num_explored), 
                             lower_bound=5, upper_bound=20
                             )
        self.explored_stc = np.random.choice(space.size(), 
                                             size=n_init_nodes, 
                                             replace=False
                                             )
        stc = space.index2gnn(self.explored_stc)
        perf = get_performance(stcs=stc, 
                               split='val', 
                               epoch=cfg.nas.visible_epoch
                               )
        perf = np.array(perf)

        self.gp.fit(x[self.explored_stc], perf)
        self.y_max = max(perf)

        for self.step in range(n_init_nodes, cfg.nas.num_explored):
            
            explored_stc_bool = np.zeros(x.size(0), dtype=np.bool_)
            explored_stc_bool[self.explored_stc] = True
            unexplored_stc_array = np.arange(x.size(0))[~explored_stc_bool]

            y_mean = self.get_utility(x[~explored_stc_bool])
            index = unexplored_stc_array[np.argmax(y_mean)]
            stc = space.index2gnn(index)
            _perf = get_performance(stc, split='val', epoch=cfg.nas.visible_epoch)

            perf = np.concatenate([perf, _perf])
            self.explored_stc = np.concatenate([self.explored_stc, [index]])
            self.gp.fit(x[self.explored_stc], perf)
            self.update_params()

            if _perf.item() > self.y_max:
                self.y_max = _perf.item()

            logger.info('[Step={}]: {} | Perf={} '.format(
                self.step, np.array(space.vars_list)[index].reshape(-1),
                round(_perf.item(), cfg.round)))
                
        return self.evaluate(space)
    