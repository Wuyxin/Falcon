import numpy as np
from graphgym.config import cfg
import graphgym.register as register


class Estimator(object):

    def __init__(self, max_epoch) -> None:
        self.max_epoch = max_epoch

    def _estimate(self, logs_by_epoch):
        raise NotImplemented

    def __call__(self, logs_by_epoch):

        if logs_by_epoch[-1]["epoch"] > self.max_epoch:
            result =  self.stored_last
        else:
            result = self._estimate(logs_by_epoch)
            self.stored_last = result
        return {"estimation": round(result, cfg.round)} 


''' 
Estimation methods (TSE-E, TSE-EMA) for NAS from
Speedy Performance Estimation for Neural Architecture Search
'''

class TSE_E(Estimator):

    name = 'tse_e'
    def __init__(self, max_epoch=50,
                 window_size=5) -> None:
        super(TSE_E, self).__init__(max_epoch)
        self.window_size = window_size
    def _estimate(self, logs_by_epoch):
        if logs_by_epoch[-1]["epoch"] < self.window_size:
            return 0
        else:
            return np.mean([logs_by_epoch[i]["loss"] \
                for i in range(self.window_size, logs_by_epoch[-1]["epoch"])])
        

class TSE_EMA(Estimator):
    
    name = 'tse_ema'
    def __init__(self, max_epoch=50,
                 decay=0.9) -> None:
        super(TSE_EMA, self).__init__(max_epoch)
        self.decay = decay
        
    def _estimate(self, logs_by_epoch):
        for i in range(logs_by_epoch[-1]["epoch"] + 1):
            if i <= 0:
                ema = logs_by_epoch[i]["loss"]
            else:
                ema = ema * (1 - self.decay) + self.decay * logs_by_epoch[i]["loss"]
        return ema


estimator_dict = {
    'tse-e': TSE_E,
    'tse-ema': TSE_EMA
}
network_dict = {**register.estimator_dict, **estimator_dict}


def create_estimators():
    estimators = []
    if cfg.estimator.name is not None:
        for name, kwargs in zip(cfg.estimator.name, cfg.estimator.kwargs):
            estimators.append(estimator_dict[name](**kwargs))
    return estimators