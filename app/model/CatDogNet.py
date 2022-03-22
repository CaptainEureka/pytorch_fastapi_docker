from random import shuffle
import glob
import os

import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Define the NN
class CatDogNet(nn.Module):
    def __init__(self):
        super(CatDogNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(50, 100, 7)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(100 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(100, 100 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x