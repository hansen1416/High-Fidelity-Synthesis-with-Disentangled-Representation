
import random
from math import sqrt

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch import distributions
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets

from tqdm.notebook import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def normal_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(mean=0, std=0.02)
        if m.bias.data is not None:
            m.bias.data.zero_()


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, c_dim=10, nc=3, infodistil_mode=False):
        super(Encoder, self).__init__()
        self.c_dim = c_dim
        self.nc = nc
        self.infodistil_mode = infodistil_mode
        self.layer = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, c_dim*2),             # B, c_dim*2
        )

    def forward(self, x):
        if self.infodistil_mode:
            x = x.add(1).div(2)
            if (x.size(2) > 64) or (x.size(3) > 64):
                x = F.adaptive_avg_pool2d(x, (64, 64))

        h = self.layer(x)
        return h


class Decoder(nn.Module):
    def __init__(self, c_dim=10, nc=3):
        super(Decoder, self).__init__()
        self.c_dim = c_dim
        self.nc = nc
        self.layer = nn.Sequential(
            nn.Linear(c_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )


    def forward(self, c):
        x = self.layer(c)
        return x