import torch
import numpy as np
from graphgym.config import cfg
from graphgym.automl.algorithms.base_meta import HrtMetaBase
from graphgym.automl.algorithms.utils.subgraph import bid_k_hop_subgraph
from graphgym.automl.algorithms.utils.tensor import to_cpu_numpy, to_tensor
from graphgym.automl.algorithms.utils.meta_stc import gen_meta_graph


class HillClimb(HrtMetaBase):

    def sample(self):
        if self.step == 0:
            indices = np.random.choice(self.meta_stc.num_nodes, size=1)
            self.status = indices
            self.status_to_check = None
        else:
            all_indices = np.arange(self.meta_stc.num_nodes)
            subset, _, _, _ = bid_k_hop_subgraph(to_tensor(self.status), 
                                                 num_hops=1, 
                                                 edge_index=self.meta_stc.edge_index
                                                 )
            unexplored_neighbors = list(set(to_cpu_numpy(subset)) - set(all_indices[self.explored]))

            if len(unexplored_neighbors) == 0:
                indices = np.random.choice(all_indices[~self.explored], size=1)
                self.status = indices
                self.status_to_check = None
            else:
                indices = np.random.choice(unexplored_neighbors, 
                                           size=min(len(unexplored_neighbors), self.sample_size), 
                                           replace=False
                                           )
                self.status_to_check = indices
        self.explored[indices] = True

        return indices
    
    def update_status(self):
        if self.status_to_check is None:
            return
        index = self.meta_stc.y[self.status_to_check].argmax()
        if self.meta_stc.y[self.status_to_check][index] > self.meta_stc.y[self.status]:
            self.status = [self.status_to_check[index]]

    def gen_meta_stc(self, space):
        return gen_meta_graph(space)