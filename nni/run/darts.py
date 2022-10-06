import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.LayerChoice([nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(3, 6, 5, padding=2)])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.LayerChoice([nn.Conv2d(6, 16, 3, padding=1), nn.Conv2d(6, 16, 5, padding=2)])
        self.conv3 = nn.Conv2d(16, 16, 1)

        self.skipconnect = nn.InputChoice(n_candidates=2)
        self.bn = nn.BatchNorm2d(16)

        self.gap = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        bs = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x0 = F.relu(self.conv2(x))
        x1 = F.relu(self.conv3(x0))

        x1 = self.skipconnect([x1, x1+x0])
        x = self.pool(self.bn(x1))

        x = self.gap(x).view(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = Net()

def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return {"acc1": (predicted == target).sum().item() / batch_size}

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii.oneshot.pytorch import DartsTrainer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)

trainer = DartsTrainer(
    model=model,
    loss=criterion,
    metrics=lambda output, target: accuracy(output, target),
    optimizer=optimizer,
    num_epochs=2,
    dataset=train_dataset,
    batch_size=64,
    log_frequency=10
    )

trainer.fit()

print('Final architecture:', trainer.export())