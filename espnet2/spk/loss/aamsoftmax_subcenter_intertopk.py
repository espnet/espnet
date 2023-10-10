#! /usr/bin/python
# -*- encoding: utf-8 -*-
# code from WeSpeaker: https://github.com/wenet-e2e/wespeaker/blob/
# c9ec537b53fe1e04525be74b2550ee95bed3a891/wespeaker/models/projections.py#L243

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class ArcMarginProduct_intertopk_subcenter(AbsLoss):
    r"""Implement of large margin arc distance with intertopk and subcenter:
    Reference:
        MULTI-QUERY MULTI-HEAD ATTENTION POOLING AND INTER-TOPK PENALTY
        FOR SPEAKER VERIFICATION.
        https://arxiv.org/pdf/2110.05042.pdf
        Sub-center ArcFace: Boosting Face Recognition by
        Large-Scale Noisy Web Faces.
        https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: norm of input feature
        margin: margin
        cos(theta + margin)
        K: number of sub-centers
        k_top: number of hard samples
        mp: margin penalty of hard samples
        do_lm: whether do large margin finetune
    """

    def __init__(
        self,
        nout,
        nclasses,
        scale=32.0,
        margin=0.2,
        easy_margin=False,
        K=3,
        mp=0.06,
        k_top=5,
        do_lm=False,
    ):
        super().__init__(nout)
        self.in_features = nout
        self.out_features = nclasses
        self.scale = scale
        self.margin = margin
        self.do_lm = do_lm

        # intertopk + subcenter
        self.K = K
        if do_lm:  # if do LMF, remove hard sample penalty
            self.mp = 0.0
            self.k_top = 0
        else:
            self.mp = mp
            self.k_top = k_top

        # initial classifier
        self.weight = nn.Parameter(torch.FloatTensor(self.K * nclasses, nout))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin
        )  # this can make the output more continuous
        ########
        self.m = self.margin
        ########
        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)

        self.ce = nn.CrossEntropyLoss()

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

        # hard sample margin is increasing as margin
        if margin > 0.001:
            mp = self.mp * (margin / 0.2)
        else:
            mp = 0.0
        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)

    def forward(self, input, label):
        if len(label.size()) == 2:
            label = label.squeeze(1)
        cosine = F.linear(
            F.normalize(input), F.normalize(self.weight)
        )  # (batch, out_dim * k)
        cosine = torch.reshape(
            cosine, (-1, self.out_features, self.K)
        )  # (batch, out_dim, k)
        cosine, _ = torch.max(cosine, 2)  # (batch, out_dim)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        if self.k_top > 0:
            # topk (j != y_i)
            _, top_k_index = torch.topk(
                cosine - 2 * one_hot, self.k_top
            )  # exclude j = y_i
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(1, top_k_index, 1)

            # sum
            output = (
                (one_hot * phi)
                + (top_k_one_hot * phi_mp)
                + ((1.0 - one_hot - top_k_one_hot) * cosine)
            )
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.ce(output, label)
        return loss
