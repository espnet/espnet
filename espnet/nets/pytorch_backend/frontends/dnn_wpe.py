from typing import Tuple

import numpy as np
import torch
import torch.nn
import torch.nn as nn
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.encoders import BRNN
from espnet.nets.pytorch_backend.encoders import BRNNP
from espnet.nets.pytorch_backend.frontends.wpe import wpe_one_iteration
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class DNN_WPE(nn.Module):
    def __init__(self,
                 wtype: str='blstmp',
                 widim: int=257,
                 wlayers: int=3,
                 wunits: int=300,
                 wprojs: int=320,
                 dropout_rate: float=0.0,
                 taps: int=5,
                 delay: int=3,
                 use_dnn_mask: bool=True,
                 iterations: int=1,
                 normalization: bool=False,
                 ):
        super().__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay

        self.normalization = normalization
        self.use_dnn_mask = use_dnn_mask

        self.inverse_power = True

        if self.use_dnn_mask:
            self.mask_est = MaskEstimator(
                wtype, widim, wlayers, wunits, wprojs, dropout_rate)

    def forward(self,
                data: ComplexTensor, ilens: torch.LongTensor) \
            -> Tuple[ComplexTensor, torch.LongTensor]:
        """

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq or Some dimension of the feature vector

        Args:
            data: (B, C, T, F)
            ilens: (B,)
        Returns:
            data: (B, C, T, F)
            ilens: (B,)
        """
        # (B, T, C, F) -> (B, F, C, T)
        enhanced = data = data.permute(0, 3, 2, 1)

        for i in range(self.iterations):
            # Calculate power: (..., C, T)
            power = enhanced.real ** 2 + enhanced.imag ** 2
            if i == 0 and self.use_dnn_mask:
                # mask: (B, F, C, T)
                mask, _ = self.mask_est(enhanced, ilens)
                if self.normalization:
                    # Normalize along T
                    mask = mask / mask.sum(dim=-1)[..., None]
                # (..., C, T) * (..., C, T) -> (..., C, T)
                power = power * mask

            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = power.mean(dim=-2)

            # enhanced: (..., C, T) -> (..., C, T)
            #   (Ignoring zero padding although it affects)
            enhanced = wpe_one_iteration(
                data.contiguous(), power,
                taps=self.taps, delay=self.delay,
                inverse_power=self.inverse_power)

            enhanced.masked_fill(make_pad_mask(ilens, enhanced.real), 0)

        # (B, F, C, T) -> (B, T, C, F)
        enhanced = enhanced.permute(0, 3, 2, 1)
        return enhanced, ilens


class MaskEstimator(torch.nn.Module):
    def __init__(self, wtype, widim, wlayers, wunits, wprojs, dropout):
        super().__init__()
        subsample = np.ones(wlayers + 1, dtype=np.int)

        if wtype == 'blstm':
            self.blstm = BRNN(widim, wlayers, wunits, wprojs, dropout)
        elif wtype == 'blstmp':
            self.blstm = BRNNP(widim, wlayers, wunits,
                                wprojs, subsample, dropout)
        else:
            raise ValueError(
                "Error: need to specify an appropriate architecture: {}"
                .format(wtype))
        self.wtype = wtype
        self.out = torch.nn.Linear(wprojs, widim)

    def forward(self, xs: ComplexTensor, ilens: torch.LongTensor) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Args:
            xs: (B, F, C, T)
            ilens: (B,)
        Returns:
            mask: (B, F, C, T)
            ilens: (B,)
        """
        # (B, F, C, T) -> (B, C, T, F)
        xs = xs.permute(0, 2, 3, 1)

        B, C, T, _ = xs.size()
        # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
        power = (xs.real ** 2 + xs.imag ** 2) ** 0.5
        xs = power ** 0.5
        # (B, C, T, F) -> (B * C, T, F)
        xs = xs.view(B * C, xs.size(-2), -1)
        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(B * C)
        xs, _ = self.blstm(xs, ilens_)
        xs = xs.view(B, C, T, -1)

        mask = self.out(xs)
        mask.masked_fill(make_pad_mask(ilens, mask, length_dim=2), 0)

        mask = torch.sigmoid(mask)

        # (B, C, T, F) -> (B, F, C, T)
        mask = mask.permute(0, 3, 1, 2)
        return mask, ilens
