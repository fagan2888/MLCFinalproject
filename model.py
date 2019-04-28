import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class CNNModel(nn.Module):
    def __init__(self, nclasses=1000):
        super(CNNModel, self).__init__()
        self.resnet = resnet50()
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, nclasses, bias=False))

    def forward(self, x):
        x = self.resnet(x)
        return x
 
