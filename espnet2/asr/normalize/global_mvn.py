from typing import Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.normalize.abs_normalization import AbsNormalization


class GlobalMVN(AbsNormalization):
    """Apply global mean and variance normalization

    Args:
        stats_file: npy file
        norm_means: Apply mean normalization
        norm_vars: Apply var normalization
        inverse: Apply inverse normalization or not
        eps:
    """

    @typechecked
    def __init__(self,
                 stats_file: str,
                 norm_means: bool = True,
                 norm_vars: bool = True,
                 inverse: bool = False,
                 eps: float = 1.0e-20):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.inverse = inverse
        self.eps = eps

        self.stats_file = stats_file
        stats = np.load(stats_file)
        count = stats[0].flatten()[-1]
        mean = stats[0, :-1] / count
        var = stats[1, :-1] / count - mean * mean
        std = np.maximum(np.sqrt(var), eps)
        self.register_buffer('mean', torch.from_numpy(mean))
        self.register_buffer('std', torch.from_numpy(std))

    def extra_repr(self):
        return f'stats_file={self.stats_file}, ' \
            f'norm_means={self.norm_means}, norm_vars={self.norm_vars}'

    def forward(self, x: torch.Tensor, ilens: torch.LongTensor = None,
                norm_means: bool = None,
                norm_vars: bool = None,
                inverse: bool = None) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)
            norm_means: Apply mean normalization
            norm_vars: Apply var normalization
            inverse: Apply inverse normalization or not

        """
        if ilens is None:
            ilens = x.new_full([x.size(0)], x.size(1))
        if norm_means is None:
            norm_means = self.norm_means
        if norm_vars is None:
            norm_vars = self.norm_vars
        if inverse is None:
            inverse = self.inverse

        self.mean = self.mean.to(x.device, x.dtype)
        self.std = self.std.to(x.device, x.dtype)
        mask = make_pad_mask(ilens, x, 1)

        if not inverse:
            # feat: (B, T, D)
            if norm_means:
                if x.is_leaf and x.requires_grad:
                    x = x - self.mean
                else:
                    x -= self.mean
            if x.is_leaf and x.requires_grad:
                x.masked_fill_(mask, 0.0)
            else:
                x.masked_fill_(mask, 0.0)

            if norm_vars:
                if x.is_leaf and x.requires_grad:
                    x = x / self.std
                else:
                    x /= self.std

            return x, ilens

        # Inverse normalize mode
        else:
            if x.is_leaf and x.requires_grad:
                x.masked_fill_(mask, 0.0)
            else:
                x.masked_fill_(mask, 0.0)

            if norm_vars:
                if x.is_leaf and x.requires_grad:
                    x = x * self.std
                else:
                    x *= self.std

            # feat: (B, T, D)
            if norm_means:
                if x.is_leaf and x.requires_grad:
                    x = x + self.mean
                else:
                    x += self.mean
                x.masked_fill_(make_pad_mask(ilens, x, 1), 0.0)
            return x, ilens
