import argparse
from typing import Dict, Optional

import torch
from typeguard import typechecked

from espnet2.uasr.segmenter.abs_segmenter import AbsSegmenter
from espnet2.utils.types import str2bool


class JoinSegmenter(AbsSegmenter):
    @typechecked
    def __init__(
        self,
        cfg: Optional[Dict] = None,
        subsample_rate: float = 0.25,
        mean_pool: str2bool = True,
        mean_join_pool: str2bool = False,
        remove_zeros: str2bool = False,
    ):
        super().__init__()

        if cfg is not None:
            cfg = argparse.Namespace(**cfg["segmentation"])
            assert cfg.type == "JOIN"
            self.subsampling_rate = cfg.subsample_rate
            self.mean_pool = cfg.mean_pool
            self.mean_pool_join = cfg.mean_pool_join
            self.remove_zeros = cfg.remove_zeros
        else:
            self.mean_pool_join = mean_join_pool
            self.remove_zeros = remove_zeros

    @typechecked
    def pre_segment(
        self,
        xs_pad: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return xs_pad, padding_mask

    @typechecked
    def logit_segment(
        self,
        logits: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        preds = logits.argmax(dim=-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        batch_size, time_length, channel_size = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )

        new_time_length = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(batch_size, new_time_length, channel_size)
        new_pad = padding_mask.new_zeros(batch_size, new_time_length)

        for b in range(batch_size):
            value, index, count = uniques[b]
            keep = value != -1

            if self.remove_zeros:
                keep.logical_and_(value != 0)

            if self.training and not self.mean_pool_join:
                value[0] = 0
                value[1:] = count.cumsum(0)[:-1]
                part = count > 1
                random = torch.rand(part.sum())
                value[part] += (count[part] * random).long()
                new_logits[b, : value.numel()] = logits[b, value]
            else:
                new_logits[b].index_add_(
                    dim=0, index=index.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : count.numel()] = new_logits[
                    b, : count.numel()
                ] / count.unsqueeze(-1).to(new_logits.device)

            new_size = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : count.numel()][keep]
                new_logits[b, :new_size] = kept_logits

            if new_size < new_time_length:
                pad = new_time_length - new_size
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True
        return new_logits, new_pad
