import torch
from torch import nn

class Network(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.num_features = num_features
        self.fc_1 = nn.Linear(self.num_features, 32)
        self.fc_2 = nn.Linear(32, 32)
        self.fc_3 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.ReLU()(self.fc_1(x))
        x = nn.ReLU()(self.fc_2(x))
        return self.fc_3(x)
