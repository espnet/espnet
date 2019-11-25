from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.normalize.abs_normalization import AbsNormalization


class GlobalMVN(AbsNormalization):
    """Apply global mean and variance normalization

    Args:
        stats_file(str): npy file
        norm_means: Apply mean normalization
        norm_vars: Apply var normalization
        std_floor(float):
    """

    @typechecked
    def __init__(self,
                 stats_file: str,
                 norm_means: bool = True,
                 norm_vars: bool = True,
                 eps: float = 1.0e-20):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars

        self.stats_file = stats_file
        stats = np.load(stats_file)
        stats = stats.astype(float)

        count = stats[0].flatten()[-1]
        mean = stats[0, :-1] / count
        var = stats[1, :-1] / count - mean * mean
        std = np.maximum(np.sqrt(var), eps)
        self.register_buffer('bias',
                             torch.from_numpy(-mean.astype(np.float32)))
        self.register_buffer('scale',
                             torch.from_numpy(1 / std.astype(np.float32)))

    def extra_repr(self):
        return f'stats_file={self.stats_file}, ' \
            f'norm_means={self.norm_means}, norm_vars={self.norm_vars}'

    def forward(self, x: torch.Tensor, ilens: torch.LongTensor) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        # feat: (B, T, D)
        if self.norm_means:
            x += self.bias.type_as(x)
            x.masked_fill(make_pad_mask(ilens, x, 1), 0.0)

        if self.norm_vars:
            x *= self.scale.type_as(x)
        return x, ilens
