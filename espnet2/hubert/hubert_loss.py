#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The HubertPretrainLoss Module uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/hubert_criterion.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

"""Hubert Pretrain Loss module."""

import torch.nn.functional as F
from torch import nn


class HubertPretrainLoss(nn.Module):
    """
        Hubert criterion module.

    This module implements the Hubert pretraining loss used for masked and
    unmasked frames. It is designed to compute the loss for both masked and
    unmasked predictions, as well as additional losses if applicable.

    Attributes:
        pred_masked_weight (float): Weight for predictive loss for masked frames.
        pred_nomask_weight (float): Weight for predictive loss for unmasked frames.
        loss_weights (float): Weights for additional loss terms (not first one).

    Args:
        pred_masked_weight (float): Weight for predictive loss for masked frames.
            Defaults to 1.0.
        pred_nomask_weight (float): Weight for predictive loss for unmasked frames.
            Defaults to 0.0.
        loss_weights (float): Weights for additional loss terms. Defaults to 10.0.

    Returns:
        Tuple[float, List[Tensor], List[Tensor]]: A tuple containing:
            - loss (float): The computed loss value.
            - logp_m_list (List[Tensor]): List of logits for masked predictions.
            - logp_u_list (List[Tensor]): List of logits for unmasked predictions.

    Raises:
        NotImplementedError: If the model does not support extra loss terms.

    Examples:
        >>> loss_module = HubertPretrainLoss()
        >>> loss, logp_m, logp_u = loss_module(model, enc_outputs)

    Note:
        This implementation utilizes code from Fairseq and is based on the work
        of Abdelrahman Mohamed and Wei-Ning Hsu.

        References:
            - Paper: https://arxiv.org/pdf/2106.07447.pdf
            - Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert
    """

    def __init__(
        self,
        pred_masked_weight: float = 1.0,
        pred_nomask_weight: float = 0.0,
        loss_weights: float = 10.0,
    ):
        super(HubertPretrainLoss, self).__init__()
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights

    def forward(self, model, enc_outputs, reduce=True):
        """
        Computes the forward pass of the Hubert pretraining loss.

        This method calculates the total loss based on the model's predictions
        for both masked and unmasked frames. It uses cross-entropy loss for the
        masked and unmasked predictions and includes additional loss terms if
        specified. The final loss is weighted according to the configured
        parameters.

        Args:
            model: The model used to obtain predictions and targets.
            enc_outputs: The encoded outputs from the model that are used to
                compute the loss.
            reduce: A boolean indicating whether to reduce the loss. If True,
                the loss will be summed; otherwise, it will not be reduced.

        Returns:
            A tuple containing:
                - loss (float): The computed loss value.
                - logp_m_list (list): The list of logits for masked frames.
                - logp_u_list (list): The list of logits for unmasked frames.

        Examples:
            >>> model = HubertModel()
            >>> enc_outputs = model.encode(inputs)
            >>> loss_fn = HubertPretrainLoss()
            >>> loss, logp_m, logp_u = loss_fn(model, enc_outputs)

        Note:
            This method assumes that the model has methods `get_logits` and
            `get_targets`, and it also checks for the existence of
            `get_extra_losses` if additional loss weights are utilized.

        Raises:
            NotImplementedError: If the model's extra losses are not a list
            containing a single element.
        """
        loss = 0.0
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
