import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.ones(1, 1, 1)

    def forward(self, x):
        channels = x.shape[1]
        self.alpha_repeat = self.alpha.repeat(1, channels, 1).to(x.device)
        return snake(x, self.alpha_repeat)
