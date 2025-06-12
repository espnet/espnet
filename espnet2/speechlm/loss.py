#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import torch


class SpeechLMCrossEntropyLossV2(torch.nn.Module):
    def __init__(
        self,
        pad,
        token_bias,
        modality_weights,
        image_interval_split,
        loss_type="mean",
        lm_head: torch.nn.Linear = None,
    ):
        super().__init__()

        self.pad = pad
        self.lm_head = lm_head
        self.loss_type = loss_type

        # (1) loss weight
        vocab_size = lm_head.weight.size(0)
        self.weight = torch.ones(vocab_size).float()
        self.weight.requires_grad_(False)
        for name, m_weight in modality_weights.items():
            if name not in token_bias:
                logging.warning(f"Modality {name} not in token_bias. Skip it")
                continue

            start, end = token_bias[name]
            self.weight[start:end] = m_weight

        # (2) aux loss interval
        self.aux_loss_interval = []
        # Codec codebook is small, put them in one interval for efficiency
        if "codec" in token_bias:
            self.aux_loss_interval.append(token_bias["codec"])
        # Image codebook is large, compute them separately
        if "image" in token_bias:
            start, end = token_bias["image"]
            assert (end - start) % image_interval_split == 0
            inc = (end - start) // image_interval_split
            i = 0
            while start + i * inc < end:
                self.aux_loss_interval.append(
                    (
                        start + i * inc,
                        start + (i + 1) * inc,
                    )
                )
                i += 1

    def forward(self, hidden, targets, loss_mask):
        # NOTE(Jinchuan): keep the weight on the correct device in the first forward.
        # We don't want to keep the weights registered as model parameters as they
        # should always be specified by external configurations.
        device, dtype = hidden[0].device, hidden[0].dtype
        self.weight = self.weight.to(device).to(dtype)

        # (1) check shape
        assert hidden.dim() == 4  # [B, T, nq, D]
        assert targets.dim() == 3  # [B, T, nq]
        assert loss_mask.dim() == 3  # [B, T, nq]
        assert hidden.size()[:3] == targets.size()
        assert hidden.size()[:3] == loss_mask.size()

        # (2) apply loss mask to targets
        targets = torch.where(loss_mask.bool(), targets, self.pad)

        elem_loss = torch.zeros_like(targets).to(dtype)
        # default prediction is never equal to a target
        acc = torch.zeros_like(targets).float() if not self.training else None

        # (3) first stream
        this_loss, this_acc, this_mask = self.forward_interval(
            hidden[:, :, 0], targets[:, :, 0]
        )
        elem_loss[:, :, 0][this_mask] = this_loss
        if this_acc is not None:
            acc[:, :, 0][this_mask] = this_acc

        # (4) all remained stream.
        # TODO: Check this is safe for single-stream LM.
        for interval in self.aux_loss_interval:
            this_loss, this_acc, this_mask = self.forward_interval(
                hidden[:, :, 1:],
                targets[:, :, 1:],
                interval=interval,
            )

            if this_loss is None:  # no target in this interval
                continue

            elem_loss[:, :, 1:][this_mask] = this_loss
            if this_acc is not None:
                acc[:, :, 1:][this_mask] = this_acc

        # (5) summarize
        weight = loss_mask[:, :, 0].sum().float()
        loss = elem_loss.sum()
        # debate on sum or mean loss: section 4.3.2 of:
        # https://arxiv.org/pdf/2411.15124
        if self.loss_type == "mean":
            loss = loss / weight

        stats = {
            "ce_loss": loss.clone().detach(),
            "weight": weight.clone().detach(),
        }

        if acc is not None:
            tok_count = loss_mask.float().sum()
            acc_all = acc.sum() / tok_count
            stats["acc_all"] = acc_all.clone().detach()

            for n in range(targets.size(2)):
                tok_count = loss_mask[:, :, n].float().sum()
                if tok_count > 0:
                    acc_layer = acc[:, :, n].sum() / tok_count
                else:
                    acc_layer = torch.Tensor([0]).to(device).float()
                stats[f"acc_layer{n}"] = acc_layer.clone().detach()

        return loss, elem_loss, stats, weight

    def forward_interval(self, hidden, targets, interval=None):
        shape = targets.size()
        hidden = hidden.flatten(end_dim=-2)
        targets = targets.flatten()

        # mask and mask select
        if interval is None:
            mask = targets != self.pad
            linear_weight = self.lm_head.weight
            weight = self.weight
            start = 0
        else:
            start, end = interval
            mask = torch.logical_and(
                targets >= start,
                targets < end,
            )
            linear_weight = self.lm_head.weight[start:end]
            weight = self.weight[start:end]

        if mask.float().sum() == 0:
            return None, None, None

        hidden, targets = hidden[mask], targets[mask]

        # loss computing
        logits = torch.matmul(hidden, linear_weight.T)
        loss = torch.nn.functional.cross_entropy(
            logits, targets - start, weight=weight, reduction="none"
        )

        if not self.training:
            acc = logits.argmax(dim=-1).eq(targets - start).float()
        else:
            acc = None

        return loss, acc, mask.view(shape)
