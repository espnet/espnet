#! /usr/bin/python
# -*- encoding: utf-8 -*-
# code from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math

class AbsLoss(nn.Module):
    def __init__(self, nOut, nClasses**kwargs):
        super().__init__()


    @abstractmethod
    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        raise NotimplementedError
