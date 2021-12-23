"""SpecAugment module with variable maximum width for time masking."""
from typing import Sequence
from typing import Union

import math

from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.mask_along_axis import MaskAlongAxis, mask_along_axis
from espnet2.layers.time_warp import TimeWarp


class SpecAug2(AbsSpecAug):
    """Implementation of SpecAug with variable maximum width for time masking.

    Reference:
        Daniel S. Park et al.
        "SpecAugment: A Simple Data
         Augmentation Method for Automatic Speech Recognition"

    .. warning::
        When using cuda mode, time_warp doesn't have reproducibility
        due to `torch.nn.functional.interpolate`.

    """

    def __init__(
        self,
        apply_time_warp: bool = True,
        time_warp_window: int = 5,
        time_warp_mode: str = "bicubic",
        apply_freq_mask: bool = True,
        freq_mask_width_range: Union[int, Sequence[int]] = (0, 27),
        num_freq_mask: int = 2,
        apply_time_mask: bool = True,
        time_mask_width_ratio_range: Union[float, Sequence[float]] = (0.0, 0.05),
        num_time_mask: int = 10,
    ):
        if not apply_time_warp and not apply_time_mask and not apply_freq_mask:
            raise ValueError(
                "Either one of time_warp, time_mask, or freq_mask should be applied",
            )
        super().__init__()
        self.apply_time_warp = apply_time_warp
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask

        if apply_time_warp:
            self.time_warp = TimeWarp(window=time_warp_window, mode=time_warp_mode)
        else:
            self.time_warp = None

        if apply_freq_mask:
            self.freq_mask = MaskAlongAxis(
                dim="freq",
                mask_width_range=freq_mask_width_range,
                num_mask=num_freq_mask,
            )
        else:
            self.freq_mask = None

        if apply_time_mask:
            self.num_time_mask = num_time_mask
            if isinstance(time_mask_width_ratio_range, float):
                time_mask_width_ratio_range = (0.0, time_mask_width_ratio_range)
            if len(time_mask_width_ratio_range) != 2:
                raise TypeError(
                    f"time_mask_width_ratio_range must be a tuple of float and float values: "
                    f"{time_mask_width_ratio_range}",
                )
            assert time_mask_width_ratio_range[1] > time_mask_width_ratio_range[0]
            self.time_mask_width_ratio_range = time_mask_width_ratio_range

    def forward(self, x, x_lengths=None):
        """Forward method of SpecAug2

        Args:
            x (torch.Tensor): (batch, length, freq)
        """
        if self.time_warp is not None:
            x, x_lengths = self.time_warp(x, x_lengths)
        if self.freq_mask is not None:
            x, x_lengths = self.freq_mask(x, x_lengths)
        if self.apply_time_mask:
            max_seq_len = x.shape[1]
            min_time_mask_width = math.floor(self.time_mask_width_ratio_range[0] * max_seq_len)
            min_time_mask_width = max([0, min_time_mask_width])
            max_time_mask_width = math.floor(self.time_mask_width_ratio_range[1] * max_seq_len)
            max_time_mask_width = min([max_seq_len, max_time_mask_width])
            if max_time_mask_width > min_time_mask_width:
                x, x_lengths = mask_along_axis(
                    spec=x,
                    spec_lengths=x_lengths,
                    mask_width_range=(min_time_mask_width, max_time_mask_width),
                    dim=1,
                    num_mask=self.num_time_mask,
                )
        return x, x_lengths
