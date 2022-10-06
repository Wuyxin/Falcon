import torch
import torchvision
from torchvision import transforms

import os
import csv

import torch.nn as nn


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import models

import os
import argparse
from run.space import *   
from run.utils.logger import Logger
from run.train_utils import train, test

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--visible_epoch', default=10, type=int)
parser.add_argument('--num_explored', default=20, type=int)
parser.add_argument('--max_epoch', default=200, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--algo', default='metagnas_labprop', type=str)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--lr', default=1e-1, type=float)
args = parser.parse_args()



device=torch.device(f'cuda:{args.device}')
os.makedirs(f'log/{args.algo}/', exist_ok=True)
logger = Logger.get_logger(name='nas', fname=f'log/{args.algo}/n={args.num_explored}-vis={args.visible_epoch}')


# Some metric functions
def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return {"acc": (predicted == target).sum().item() / batch_size}

def element_wise_acc(output, target):
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).float().view(-1)


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = torchvision.datasets.CIFAR10(
    root='data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root='data', train=False, download=True, transform=transform_test)

# Search
net = models.__dict__[args.model]()
net.cuda()
print(torch.cuda.device_count())
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
print('Using CUDA..')
hypo_space = {
    'lr_decay': [0.1, 0.2, 0.5, 0.8], 
    'momentum': [0.5, 0.9, 0.99], 
    'weight_decay': [1e-4, 5e-4, 1e-3, 5e-3], 
    'patience': [3, 5, 10], 
    'batch_size': [32, 64, 128, 256]}

criterion = nn.CrossEntropyLoss()
if args.algo == 'metagnas_labprop':
    from nni.retiarii.strategy import FalconTrainer
    trainer = FalconTrainer(
        stc_cls=models.__dict__[args.model],
        mutated_stc=net,
        hypo_space=hypo_space,
        loss=criterion,
        meta_metric=element_wise_acc, 
        metrics=lambda output, target: accuracy(output, target),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        workers=args.workers,
        resume_dir='param/',
        device=device
        )

    stc, sel_perf, test_perf = trainer.run(visible_epoch=args.visible_epoch, 
                                num_explored=args.num_explored, 
                                max_epoch=args.max_epoch)
    result = {key: int(value) for key, value in zip(trainer.variables, stc)} 
