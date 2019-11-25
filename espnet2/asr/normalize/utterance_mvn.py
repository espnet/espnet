from typing import Tuple

import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.normalize.abs_normalization import AbsNormalization


class UtteranceMVN(AbsNormalization):
    @typechecked
    def __init__(self,
                 norm_means: bool = True,
                 norm_vars: bool = False,
                 eps: float = 1.0e-20):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps

    def extra_repr(self):
        return f'norm_means={self.norm_means}, norm_vars={self.norm_vars}'

    def forward(self, x: torch.Tensor, ilens: torch.LongTensor) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        return utterance_mvn(x, ilens,
                             norm_means=self.norm_means,
                             norm_vars=self.norm_vars,
                             eps=self.eps)


def utterance_mvn(
        x: torch.Tensor,
        ilens: torch.LongTensor,
        norm_means: bool = True,
        norm_vars: bool = False,
        eps: float = 1.0e-20) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Apply utterance mean and variance normalization

    Args:
        x: (B, T, D), assumed zero padded
        ilens: (B, T, D)
        norm_means:
        norm_vars:
        eps:

    """
    ilens_ = ilens.type_as(x)
    # mean: (B, D)
    mean = x.sum(dim=1) / ilens_[:, None]

    if norm_means:
        x -= mean[:, None, :]
        x_ = x
    else:
        x_ = x - mean[:, None, :]

    # Zero padding
    x_.masked_fill(make_pad_mask(ilens, x_, 1), 0.0)
    if norm_vars:
        var = x_.pow(2).sum(dim=1) / ilens_[:, None]
        var = torch.clamp(var, min=eps)
        x /= var.sqrt()[:, None, :]
        x_ = x
    return x_, ilens
