import math

import torch
from typeguard import typechecked

from espnet2.uasr.segmenter.abs_segmenter import AbsSegmenter
from espnet2.utils.types import str2bool


class RandomSegmenter(AbsSegmenter):
    @typechecked
    def __init__(
        self,
        subsample_rate: float = 0.25,
        mean_pool: str2bool = True,
        mean_join_pool: str2bool = False,
        remove_zeros: str2bool = False,
    ):
        super().__init__()
        self.subsample_rate = subsample_rate

    def pre_segment(
        self,
        xs_pad: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_num = math.ceil(xs_pad.size(1) * self.subsample_rate)
        ones = torch.ones(xs_pad.shape[:-1], device=xs_pad.device)
        indices, _ = ones.multinomial(target_num).sort(dim=-1)
        indices_ld = indices.unsqueeze(-1).expand(-1, -1, xs_pad.size(-1))
        xs_pad = xs_pad.gather(1, indices_ld)
        padding_mask = padding_mask.gather(1, index=indices)
        return xs_pad, padding_mask

    def logit_segment(
        self,
        xs_pad: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return xs_pad, padding_mask
