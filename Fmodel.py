import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpNet(nn.Module):

    def __init__(self):
        super(InterpNet, self).__init__()
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
