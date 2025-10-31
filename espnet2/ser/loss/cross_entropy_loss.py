#! /usr/bin/python
# -*- encoding: utf-8 -*-
# code from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch.nn as nn

from espnet2.ser.loss.abs_loss import AbsLoss


class Xnt(AbsLoss):
    def __init__(self, nout, nclasses, **kwargs):
        super().__init__(nout)

        self.in_feats = nout
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        loss = self.ce(x, label)
        return loss
