from typing import Tuple

from pytorch_wpe import wpe_one_iteration
import torch
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class DNN_WPE(torch.nn.Module):
    def __init__(
        self,
        wtype: str = "blstmp",
        widim: int = 257,
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        num_spkr: int = 1,
        dropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask: bool = True,
        iterations: int = 1,
        normalization: bool = False,
    ):
        super(DNN_WPE, self).__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay
        self.num_spkr = num_spkr

        self.normalization = normalization
        self.use_dnn_mask = use_dnn_mask

        self.inverse_power = True

        if self.use_dnn_mask:
            self.mask_est = MaskEstimator(
                wtype, widim, wlayers, wunits, wprojs, dropout_rate, nmask=num_spkr
            )

    def forward(
        self, data: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function

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
        def dereverberate(data, power, iterations, taps, delay, inverse_power):
            enhanced = data
            for i in range(iterations):
                if i > 0:
                    power = enhanced.real ** 2 + enhanced.imag ** 2
                    # Averaging along the channel axis: (..., C, T) -> (..., T)
                    power = power.mean(dim=-2)
                # enhanced: (..., C, T) -> (..., C, T)
                enhanced = wpe_one_iteration(
                    data.contiguous(),
                    power,
                    taps=taps,
                    delay=delay,
                    inverse_power=inverse_power,
                )
            return enhanced

        # (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)
        # Calculate power: (..., C, T)
        power = data.real ** 2 + data.imag ** 2

        if self.use_dnn_mask:
            mask, _ = self.mask_est(data, ilens)
            if self.normalization:
                # Normalize along T
                mask = [m / m.sum(dim=-1)[..., None] for m in mask]
            power = [(power * m).mean(dim=-2) for m in mask]
        else:
            mask = None
            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = [power.mean(dim=-2)] * self.num_spkr

        if self.num_spkr == 1:
            enhanced = dereverberate(
                data, power[0], self.iterations, self.taps, self.delay, self.inverse_power
            )
            enhanced.masked_fill_(make_pad_mask(ilens, enhanced.real), 0)
            # (B, F, C, T) -> (B, T, C, F)
            enhanced = enhanced.permute(0, 3, 2, 1)

            if mask is not None:
                mask = mask.transpose(-1, -3)
        else:
            # multi-speaker case: (mask_speech1, mask_speech2, ...)
            enhanced = [
                dereverberate(
                    data, power[spk], self.iterations, self.taps, self.delay, self.inverse_power
                ) for spk in range(self.num_spkr)
            ]
            for enh in enhanced:
                enh.masked_fill_(make_pad_mask(ilens, enh.real), 0)
            # (B, F, C, T) -> (B, T, C, F)
            enhanced = [enh.permute(0, 3, 2, 1) for enh in enhanced]

            if mask is not None:
                mask = [m.transpose(-1, -3) for m in mask]

        return enhanced, ilens, mask
