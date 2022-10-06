import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.nas.pytorch.search_space_zoo import ENASMacroGeneralModel
from nni.nas.pytorch.search_space_zoo.enas_ops import ConvBranch, PoolBranch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        in_filters = 3
        out_filters = 6
        self.conv1 = nn.LayerChoice([
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=False),
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=True),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=False),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=True),
            PoolBranch('avg', in_filters, out_filters, 3, 1, 1),
            PoolBranch('max', in_filters, out_filters, 3, 1, 1)], label='conv1')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = nn.Conv2d(out_filters, out_filters, 1)
        self.skipconnect1_1 = nn.InputChoice(n_candidates=2, label='skipconnect1_1')
        
        in_filters = 6
        out_filters = 16
        self.conv2 = nn.LayerChoice([
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=False),
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=True),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=False),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=True),
            PoolBranch('avg', in_filters, out_filters, 3, 1, 1),
            PoolBranch('max', in_filters, out_filters, 3, 1, 1)], label='conv2')
        self.conv2_1 = nn.Conv2d(out_filters, out_filters, 1)
        self.skipconnect2_1 = nn.InputChoice(n_candidates=2, label='skipconnect2_1')
        self.bn = nn.BatchNorm2d(out_filters)

        self.gap = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        bs = x.size(0)
        x0 = self.pool(F.relu(self.conv1(x)))
        x1 = F.relu(self.conv1_1(x0))
        x = self.skipconnect1_1([x1, x1+x0])

        x0 = F.relu(self.conv2(x))
        x1 = F.relu(self.conv2_1(x0))
        x1 = self.skipconnect2_1([x1, x1+x0])
        x = self.pool(self.bn(x1))

        x = self.gap(x).view(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return {"acc": (predicted == target).sum().item() / batch_size}

def element_wise_acc(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).float().view(-1)

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii import serialize
import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii.strategy import RandomDummyTrainer

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = serialize(CIFAR10, root="data", train=True, download=True, transform=transform)
test_dataset = serialize(CIFAR10, root="data", train=False, download=True, transform=transform)


model = Net()
criterion = torch.nn.CrossEntropyLoss()
trainer = RandomDummyTrainer(
    stc_cls=Net,
    mutated_stc=model,
    loss=criterion,
    meta_metric=element_wise_acc, 
    metrics=lambda output, target: accuracy(output, target),
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    resume_dir='param/',
    device=torch.device('cuda')
    )

trainer.run()

print('Final architecture:', trainer.export())