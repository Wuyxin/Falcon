from re import X
import torch
import time
import logging

from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch_data in loader:
        batch_data.split = 'train'
        optimizer.zero_grad()
        batch_data.to(torch.device(cfg.device))

        x, edge_index = batch_data.x, batch_data.edge_index
        edge_attr = batch_data.edge_attr if hasattr(batch_data, 'edge_attr') else None
        batch = batch_data.batch if hasattr(batch_data, 'batch') else None
        pred = model(x, edge_index, edge_attr, batch)
        y = batch_data.y
        try:
            mask = batch_data.train_mask
            pred, y = pred[mask], y[mask]
            loss, pred_score = compute_loss(pred, y)
        except:
            loss, pred_score = compute_loss(pred, y)
            
        loss.backward()
        optimizer.step()
        logger.update_stats(true=y.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch_data in loader:
        batch_data.split = split
        batch_data.to(torch.device(cfg.device))

        x, edge_index = batch_data.x, batch_data.edge_index
        edge_attr = batch_data.edge_attr if hasattr(batch_data, 'edge_attr') else None
        batch = batch_data.batch if hasattr(batch_data, 'batch') else None
        pred = model(x, edge_index, edge_attr, batch)
        y = batch_data.y
        try:
            mask = eval(f'batch_data.{split}_mask')
            pred, y = pred[mask], y[mask]
            loss, pred_score = compute_loss(pred, y)
        except:
            loss, pred_score = compute_loss(pred, y)

        logger.update_stats(true=y.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def train(loggers, loaders, model, optimizer, scheduler, estimators=None):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch, estimators)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
