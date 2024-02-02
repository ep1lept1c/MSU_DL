import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.modules.batchnorm import BatchNorm1d

# Реализуйте модель.

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024, bias=True)
        self.bn1 = BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256, bias=True)
        self.bn2 = BatchNorm1d(256)
        self.fc_out = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x