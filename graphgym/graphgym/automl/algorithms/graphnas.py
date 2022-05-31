'''
Adapted from https://github.com/GraphNAS/GraphNAS
'''
import os
import glob
import logging
import scipy.signal
import numpy as np

import torch

from graphgym.config import cfg
from graphgym.automl.algorithms.base import Base
from graphgym.automl.estimate.scratch import get_performance
from graphgym.automl.algorithms.utils.tensor import get_variable, to_item
from graphgym.automl.algorithms.utils.graphnas_controller import SimpleNASController

logger = logging.getLogger('nas')
history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value


class TopAverage(object):
    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_top_average(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return 0

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[:self.top_k]
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)


class GraphNASMacro(Base):
    
    def __init__(self, args, **kwargs):
        super(GraphNASMacro, self).__init__(args, **kwargs)
        self.controller_step = 0  # counter for controller
        self.submodel_manager = None
        self.controller = None
        self.reward_manager = TopAverage(10)
        # For budgeted nas 
        unit = cfg.nas.num_explored // cfg.graphnas.max_inner_loop
        assert unit > 0
        if unit < cfg.graphnas.max_outer_loop:
            self.batch_size = 1
            cfg.graphnas.max_outer_loop = unit
        else:
            self.batch_size = max(1, unit // cfg.graphnas.max_outer_loop)
        
    def init_model(self, space):
        # generate model description in macro way (generate entire network description)
        self.search_space = space.search_space
        action_names = list(self.search_space.keys())
        self.action_list = action_names * cfg.graphnas.layers_of_child_model
        # build RNN controller
        self.controller = SimpleNASController(action_list=self.action_list,
                                              search_space=self.search_space,
                                              cuda=cfg.device
                                              )
        self.controller.to(cfg.device)


    def search(self, space, **kwargs):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """
        self.space = space
        self.init_model(space)  # build controller and sub-model
        controller_optimizer = torch.optim.Adam
        self.controller_optim = controller_optimizer(self.controller.parameters(), 
                                                    lr=cfg.graphnas.lr)
            
        for self.step in range(cfg.graphnas.max_outer_loop):
            # Training the controller parameters theta
            self.train_controller()
            # Derive architectures
            best_actions = self.derive(sample_num=cfg.nas.n_fully_train)
        
        try: 
            self.space.vars2gnn(best_actions)
        except:
            return None, torch.tensor([-1]), torch.tensor([-1])
        logger.info("best structure:" + str(best_actions))
        return self.space.vars2gnn(best_actions),\
               self.evaluate(best_actions, split='val', epoch=cfg.optim.max_epoch),\
               self.evaluate(best_actions, split='test', epoch=cfg.optim.max_epoch)


    def get_reward(self, gnn_list, entropies, hidden):
        """
        Computes the reward of a single sampled model on validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        if isinstance(gnn_list, dict):
            gnn_list = [gnn_list]
        if isinstance(gnn_list[0], list) or isinstance(gnn_list[0], dict):
            pass
        else:
            gnn_list = [gnn_list]  # when structure_list is one structure

        reward_list = []
        for gnn in gnn_list:
            val_perf = self.evaluate(gnn, split='val', epoch=cfg.nas.visible_epoch)
            val_perf = val_perf.item()
            reward = self.reward_manager.get_reward(val_perf)
            reward_list.append(reward)
        reward_list = np.array(reward_list).reshape(-1, 1)
        
        if cfg.graphnas.entropy_mode == 'reward':
            rewards = reward_list + cfg.graphnas.entropy_coeff * entropies
        elif cfg.graphnas.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {cfg.graphnas.entropy_mode}')

        return rewards, hidden

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        model = self.controller
        model.train()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.controller.init_hidden(self.batch_size)
        loss_logger = []
        for _ in range(cfg.graphnas.max_inner_loop):
            # sample graphnas
            structure_list, log_probs, entropies = self.controller.sample(self.batch_size, with_details=True)
            assert len(structure_list) == self.batch_size

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            results = self.get_reward(structure_list, np_entropies, hidden)

            if results:  # has reward
                rewards, hidden = results
            else:
                continue  # CUDA Error happens, drop structure and step into next iteration

            if 1 > cfg.graphnas.discount > 0:
                rewards = scipy.signal.lfilter(
                    [1], [1, -cfg.graphnas.discount], rewards[::-1], axis=0
                    )[::-1]

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = cfg.graphnas.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=0.5)
            adv_history.extend(adv)

            adv = get_variable(adv, cfg.device, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            if cfg.graphnas.entropy_mode == 'regularizer':
                loss -= cfg.graphnas.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if cfg.graphnas.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              cfg.graphnas.controller_grad_clip)
            self.controller_optim.step()
            loss_logger.append(to_item(loss.data))
            self.controller_step += 1
        
        logger.info('** training controller: loss={}'.format(round(np.mean(loss_logger), cfg.round)))

    def evaluate(self, action, epoch, split='val'):
        """
        Evaluate a structure on the validation set.
        """
        try:
            self.controller.eval()
            gnn = self.space.vars2gnn(action)
            perf = get_performance(gnn, split=split, epoch=epoch)
        except FileNotFoundError:
            # assign the evaluate as -1 if the structure is invalid
            perf = -torch.ones(1)
        return perf

    def derive(self, sample_num):
        """
        sample a serial of structures, and return the best structure.
        """

        max_R = 0
        best_actions = None
        gnn_list, _, _ = self.controller.sample(sample_num, with_details=True)
        for action in gnn_list:
            results = self.evaluate(action, split='val', epoch=cfg.nas.visible_epoch)
            results = results.item()
            if results > max_R:
                max_R = results
                best_actions = action

        logger.info(f'derive | action:{best_actions} | max_R: {max_R:8.6f}')

        return best_actions

        