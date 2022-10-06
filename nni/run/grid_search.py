import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
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


import nni.retiarii.strategy as strategy

simple_strategy = strategy.GridSearch() # choice: Random, GridSearch, RegularizedEvolution, TPEStrategy

from torchvision import transforms
from torchvision.datasets import CIFAR10
from nni.retiarii import serialize
import nni.retiarii.evaluator.pytorch.lightning as pl

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = serialize(CIFAR10, root="data", train=True, download=True, transform=transform)
test_dataset = serialize(CIFAR10, root="data", train=False, download=True, transform=transform)

trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=64),
                            val_dataloaders=pl.DataLoader(test_dataset, batch_size=64),
                            max_epochs=2, gpus=[7])


from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment

exp = RetiariiExperiment(model, trainer, [], simple_strategy)

exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'Retiarii example'
exp_config.trial_concurrency = 2
exp_config.max_trial_number = 1
exp_config.trial_gpu_number = 2
exp_config.max_experiment_duration = '5m'
exp_config.execution_engine = 'base'
exp_config.training_service.use_active_gpu = True


exp.run(exp_config, 8745)


print('Final model:')
for model_code in exp.export_top_models():
    print(model_code)