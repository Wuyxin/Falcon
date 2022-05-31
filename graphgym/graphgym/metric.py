from asyncio.log import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from graphgym.config import cfg
from graphgym.logger_pyg import infer_task
from sklearn.metrics import *

def get_pred_int(pred_score):
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            return (pred_score > cfg.model.thresh).long()
        else:
            return pred_score.max(dim=1)[1]


def compute_metric(pred_score, true):
    '''
    :param pred: unnormalized prediction
    :param true: label
    :return: predictive metric
    '''
    pred_int = get_pred_int(pred_score)
    task_type = infer_task()
    if task_type == 'classification_binary':
        try:
            return {'auc': round(roc_auc_score(true, pred_int), cfg.round)}
        except:
            logger.info("Run into error when computing ROC-AUC")
            return {'auc': 0.0}
    elif task_type == 'classification_multi' or \
        task_type == 'link_pred':
        return {'accuracy': round(accuracy_score(true, pred_int), cfg.round)}
    elif task_type == 'regression':
        return {'mse': float(round(mean_squared_error(true, pred_int), cfg.round))}
    elif task_type == 'multi_tasks_classification':
        true, pred_score = torch.cat(true), torch.cat(pred_int)
        rocauc_list = []
        for i in range(true.shape[1]):
            if torch.sum(true[:,i] == 1) > 0 and torch.sum(true[:,i] == 0) > 0:
                is_labeled = true[:,i] == true[:,i]
                # AUC is only defined when there is at least one positive data.
                if torch.sum(true[is_labeled,i] == true[is_labeled,i][0]) == true[is_labeled,i].numel():
                    print("Fail computing auc. Model predictions are all the same!")
                    auc = 0
                else:
                    auc = roc_auc_score(true[is_labeled,i], pred_score[is_labeled,i])
                rocauc_list.append(auc)

        return {'auc': round(sum(rocauc_list)/len(rocauc_list), cfg.round)}


def compute_instance_metric(pred_score, true):
    '''
    :param pred: unnormalized prediction
    :param true: label
    :return: predictive metric
    '''
    # print('pred_score', pred_score.size())
    # print('true', true.size())
    pred_int = get_pred_int(pred_score)
    task_type = infer_task()
    if task_type == 'classification_binary' or\
        task_type == 'classification_multi' or \
        task_type == 'link_pred':
        return pred_int.view(-1) == true.view(-1)
    return None

    # elif task_type == 'regression':
    #     return {'mse': float(round(mean_squared_error(true, pred_int), cfg.round))}
    # elif task_type == 'multi_tasks_classification':
    #     true, pred_score = torch.cat(true), torch.cat(pred_int)
    #     rocauc_list = []
    #     for i in range(true.shape[1]):
    #         if torch.sum(true[:,i] == 1) > 0 and torch.sum(true[:,i] == 0) > 0:
    #             is_labeled = true[:,i] == true[:,i]
    #             # AUC is only defined when there is at least one positive data.
    #             if torch.sum(true[is_labeled,i] == true[is_labeled,i][0]) == true[is_labeled,i].numel():
    #                 print("Fail computing auc. Model predictions are all the same!")
    #                 auc = 0
    #             else:
    #                 auc = roc_auc_score(true[is_labeled,i], pred_score[is_labeled,i])
    #             rocauc_list.append(auc)

    #     return {'auc': round(sum(rocauc_list)/len(rocauc_list), cfg.round)}