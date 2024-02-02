import numpy as np
import torch
from torch import nn

def create_model():
    model_layer = nn.Sequential(nn.Linear(784, 256, bias=True),
                   nn.ReLU(),
                   nn.Linear(256, 16, bias=True),
                   nn.ReLU(),
                   nn.Linear(16, 10, bias=True))
    return model_layer


def count_parameters(model):
    count = 0
    for param in model.parameters():
        count += param.numel()
    return count