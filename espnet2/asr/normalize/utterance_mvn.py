from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.normalize.abs_normalization import AbsNormalization


class UtteranceMVN(AbsNormalization):
    def __init__(self,
                 norm_means: bool = True,
                 norm_vars: bool = False,
                 eps: float = 1.0e-20):
        assert check_argument_types()
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps

    def extra_repr(self):
        return f'norm_means={self.norm_means}, norm_vars={self.norm_vars}'

    def forward(self, x: torch.Tensor, ilens: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)

        """
        return utterance_mvn(x, ilens,
                             norm_means=self.norm_means,
                             norm_vars=self.norm_vars,
                             eps=self.eps)


def utterance_mvn(
        x: torch.Tensor,
        ilens: torch.Tensor = None,
        norm_means: bool = True,
        norm_vars: bool = False,
        eps: float = 1.0e-20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply utterance mean and variance normalization

    Args:
        x: (B, T, D), assumed zero padded
        ilens: (B,)
        norm_means:
        norm_vars:
        eps:

    """
    if ilens is None:
        ilens = x.new_full([x.size(0)], x.size(1))
    ilens_ = ilens.to(x.device, x.dtype).view(-1,
                                              *[1 for _ in range(x.dim() - 1)])
    # mean: (B, D)
    mean = x.sum(dim=1, keepdim=True) / ilens_

    if norm_means:
        if x.is_leaf and x.requires_grad:
            x -= mean
        else:
            x = x - mean

        # Zero padding
        x.masked_fill_(make_pad_mask(ilens, x, 1), 0.0)

        if norm_vars:
            var = x.pow(2).sum(dim=1, keepdim=True) / ilens_
            std = torch.clamp(var.sqrt(), min=eps)
            x /= std.sqrt()
        return x, ilens
    else:
        if x.is_leaf and x.requires_grad:
            x = x.masked_fill(make_pad_mask(ilens, x, 1), 0.0)
        else:
            x.masked_fill_(make_pad_mask(ilens, x, 1), 0.0)
        if norm_vars:
            var = (x - mean).pow(2).sum(dim=1, keepdim=True) / ilens_
            std = torch.clamp(var.sqrt(), min=eps)
            if x.is_leaf and x.requires_grad:
                x = x / std.sqrt()
            else:
                x /= std
        return x, ilens

