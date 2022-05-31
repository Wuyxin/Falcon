import numpy as np
from graphgym.config import cfg
from graphgym.automl.algorithms.base import Base


class Random(Base):

    def search(self, space, **kwargs):
        self.explored_stc = np.random.choice(space.size(), 
                                             size=cfg.nas.num_explored, 
                                             replace=False
                                            )

        return self.evaluate(space, split='val')