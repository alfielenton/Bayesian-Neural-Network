import torch
from torch import nn

class Network(nn.Module):

    def __init__(self, num_channels, num_classes, num_med_channels=64):

        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_med_channels = num_med_channels

        self.conv_layers = nn.Sequential(nn.Conv2d(self.num_channels, self.num_med_channels, kernel_size=6, stride=2),
                                                  nn.ReLU(), 
                                                  nn.Conv2d(self.num_med_channels, self.num_med_channels, kernel_size=4, stride=2),
                                                  nn.ReLU(), 
                                                  nn.Conv2d(self.num_med_channels, self.num_med_channels, kernel_size=4, stride=2), 
                                                  nn.ReLU(), 
                                                  nn.MaxPool2d(kernel_size=3, stride=2), 
                                                  nn.MaxPool2d(kernel_size=3, stride=1))

        self.fc_layers = nn.Sequential(nn.Linear(9 * 9 * self.num_med_channels, 2048), 
                                       nn.ReLU(), 
                                       nn.Linear(2048, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, self.num_classes))

    def forward(self, x):

        x = self.conv_layers(x)
        x = x.view(-1, 9 * 9 * self.num_med_channels)
        return self.fc_layers(x)