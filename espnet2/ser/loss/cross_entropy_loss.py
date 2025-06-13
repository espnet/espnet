#! /usr/bin/python
# -*- encoding: utf-8 -*-
# code from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.ser.loss.abs_loss import AbsLoss


class Xnt(AbsLoss):
    """Additive angular margin softmax.

    Paper: Deng, Jiankang, et al. "Arcface: Additive angular margin loss for
    deep face recognition." Proceedings of the IEEE/CVF conference on computer
    vision and pattern recognition. 2019.

    args:
        nout    : dimensionality of speaker embedding
        nclases: number of speakers in the training set
    """

    def __init__(self, nout, nclasses, **kwargs):
        super().__init__(nout)

        self.in_feats = nout
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        # if len(label.size()) == 2:
        #     label = label.squeeze(1)

        # assert x.size()[0] == label.size()[0]
        # assert x.size()[1] == self.in_feats

        loss = self.ce(x, label)
        return loss
