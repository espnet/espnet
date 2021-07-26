#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""

import torch
from torch import nn
import torch.nn.functional as F


class HubertPretrainLoss(nn.Module):
    """

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(
        self,
        pred_masked_weight: float = 1.0,
        pred_nomask_weight: float = 0.0,
        loss_weights: float = 10.0,
    ):
        """Construct an LabelSmoothingLoss object."""
        super(HubertPretrainLoss, self).__init__()
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights

    def forward(self, model, enc_outputs, reduce=True):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        loss = 0.
        sample_size = 0
        reduction = "sum" if reduce else "none"
        
        loss_m_list = []
        logp_m_list = model.get_logits(enc_outputs, True)
        targ_m_list = model.get_targets(enc_outputs, True)
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
            loss_m_list.append(loss_m)
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += targ_m_list[0].numel()

        loss_u_list = []
        logp_u_list = model.get_logits(enc_outputs, False)
        targ_u_list = model.get_targets(enc_outputs, False)
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += targ_u_list[0].numel()

        if self.loss_weights > 0:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(enc_outputs)

            if isinstance(extra_losses, list):
                extra_losses = extra_losses[0]
                names = names[0]
            else:
                raise NotImplementedError("only support one extra loss")
            loss += self.loss_weights * extra_losses.float() * sample_size

        return loss, logp_m_list, logp_u_list


