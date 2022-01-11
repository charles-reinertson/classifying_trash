
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: define each layer
        # in_channel=3 for (Red,Green,Blue)
        # out_channel=16 for # of filters
        # kernel_size=Filter size
        # stride = stride
        # padding = padding
        # bias = bias
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=(2, 2))
        self.fc1 = nn.Linear(32768, 5000)
        self.fc2 = nn.Linear(5000, 500)
        self.fc3 = nn.Linear(500, 60)
        #

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)

        # TODO: initialize the parameters for [self.fc1, self.fc2, self.fc3]
        for fc in [self.fc1, self.fc2, self.fc3]:
            F_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(F_in))
            nn.init.constant_(fc.bias, 0.0)
        #

    def forward(self, x):
        N, C, H, W = x.shape

        # TODO: forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
