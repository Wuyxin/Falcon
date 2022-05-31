import numpy as np
from graphgym.automl.algorithms.base_meta import HrtMetaBase
from graphgym.automl.algorithms.utils.meta_stc import gen_meta_graph
from graphgym.automl.algorithms.utils.subgraph import bid_k_hop_subgraph
from graphgym.automl.algorithms.utils.tensor import to_cpu_numpy, to_tensor


class SimulatedAnneal(HrtMetaBase):

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
            neighbors = set(to_cpu_numpy(subset)) - set(all_indices[self.explored])
            neighbors = list(neighbors)

            if len(neighbors) == 0:
                indices = np.random.choice(all_indices[~self.explored], size=1)
                self.status = indices
                self.status_to_check = None
            else:
                indices = np.random.choice(neighbors, 
                                           size=min(len(neighbors), self.sample_size), 
                                           replace=False
                                          )
                self.status_to_check = indices
        self.explored[indices] = True
        return indices

    def update_status(self, tau=1):
        if self.status_to_check is None:
            return
        y_check = self.meta_stc.y[self.status_to_check]
        index = y_check.argmax()
        if y_check[index] > self.meta_stc.y[self.status]:
            self.status = [self.status_to_check[index]]
        else:
            delta = self.meta_stc.y[self.status] - max(y_check)
            delta = delta.detach().cpu().numpy()
            temp = tau / self.step 
            p_max = np.exp(-delta / temp).item() 

            if np.random.choice([0, 1], p=[1 - p_max, p_max]):
                self.status = [self.status_to_check[index]]

    def gen_meta_stc(self, space):
        return gen_meta_graph(space)