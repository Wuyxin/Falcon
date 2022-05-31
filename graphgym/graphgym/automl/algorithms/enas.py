import torch
import logging
import json
import os
from nni.retiarii import fixed_arch

from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.automl.estimate.scratch import get_performance_from_log
from graphgym.train_mutable import train
from graphgym.metric import compute_metric
from graphgym.loader_pyg import create_loader
from graphgym.logger_pyg import create_logger, setup_printing
from graphgym.estimator import create_estimators
from graphgym.optimizer import create_optimizer, create_scheduler
from graphgym.utils.comp_budget import params_count
from graphgym.utils.agg_runs import agg_runs
from graphgym.automl.algorithms.base import Base
from graphgym.models.gnn_mutable import GNN
from graphgym.automl.algorithms.nni_drive.enas import EnasTrainer

logger = logging.getLogger('nas')

def metric_as_reward(output, target):
    return torch.tensor(
        list(compute_metric(output, target).values()))

class Enas(Base):
    def __init__(self, args, reward_function=None, **kwargs):
        super(Enas, self).__init__(args, **kwargs)
        if reward_function is None:
            self.reward_function = metric_as_reward
        else:
            self.reward_function = reward_function

    def search(self, space, **kwargs):
        
        # loader dataset 
        cfg.merge_from_file(cfg.nas.config_base)
        self.loaders = create_loader()
        train_dataset = self.loaders[0].dataset
        val_dataset = self.loaders[1].dataset
        self.space_dict = space.space_dict

        # build mutable model
        self.model = GNN(dim_in=cfg.share.dim_in,
                         dim_out=cfg.share.dim_out,
                         space_dict=self.space_dict
                         )
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=cfg.enas.lr, 
                                    momentum=0.9
                                    )
        logger.info(self.model)
        # enas training, supported by nni
        self.trainer = EnasTrainer(model=self.model,
                                   loss=compute_loss,
                                   metrics=compute_metric,
                                   reward_function=self.reward_function,
                                   optimizer=optimizer,
                                   num_epochs=cfg.enas.epoch,
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   batch_size=cfg.enas.batch_size,
                                   log_frequency=10,
                                   device=cfg.device
                                   )
        self.trainer.fit()
        return self.evaluate(space)
    
    def evaluate(self, space):

        os.makedirs(os.path.join(cfg.nas.resume_dir, 'enas'), exist_ok=True)
        stc_checkpoint = os.path.join(cfg.nas.resume_dir, 'enas',
                                      f'ckp-seed={cfg.seed}-epoch={cfg.enas.epoch}.json')
        best_stc = self.trainer.export()
        json.dump(self.trainer.export(), open(stc_checkpoint, 'w'))
        with fixed_arch(stc_checkpoint):
            self.fixed_model = GNN(dim_in=cfg.share.dim_in,
                                   dim_out=cfg.share.dim_out,
                                   space_dict=self.space_dict
                                   )

        cfg.params = params_count(self.fixed_model)
        self.fixed_model.to(cfg.device)
        setup_printing()
        loggers = create_logger()
        estimators = create_estimators()
        optimizer = create_optimizer(self.fixed_model.parameters())
        scheduler = create_scheduler(optimizer)
        train(loggers, self.loaders, self.fixed_model, optimizer, scheduler, estimators)

        print(best_stc)
        from pathlib import Path
        parent_dir = Path(cfg.out_dir).parent.absolute()
        agg_runs(parent_dir, cfg.metric_best)
        
        logger.info(f'Best models: {self.fixed_model} \n {best_stc}')
        val_perf = get_performance_from_log(parent_dir, split='val', epoch=cfg.optim.max_epoch)
        test_perf = get_performance_from_log(parent_dir, split='test', epoch=cfg.optim.max_epoch)
        return best_stc, torch.tensor(val_perf), torch.tensor(test_perf)

    