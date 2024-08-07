"""SpecAugment module."""

from typing import Optional, Sequence, Union

from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.mask_along_axis import MaskAlongAxis, MaskAlongAxisVariableMaxWidth
from espnet2.layers.time_warp import TimeWarp


class SpecAug(AbsSpecAug):
    """
        Implementation of SpecAugment for speech augmentation.

    This class applies various augmentation techniques to speech spectrograms,
    including time warping, frequency masking, and time masking.

    Attributes:
        apply_time_warp (bool): Whether to apply time warping.
        apply_freq_mask (bool): Whether to apply frequency masking.
        apply_time_mask (bool): Whether to apply time masking.
        time_warp (TimeWarp): Time warping object if enabled, else None.
        freq_mask (MaskAlongAxis): Frequency masking object if enabled, else None.
        time_mask (Union[MaskAlongAxis, MaskAlongAxisVariableMaxWidth]): Time masking object if enabled, else None.

    Args:
        apply_time_warp (bool): Whether to apply time warping. Defaults to True.
        time_warp_window (int): Window size for time warping. Defaults to 5.
        time_warp_mode (str): Interpolation mode for time warping. Defaults to "bicubic".
        apply_freq_mask (bool): Whether to apply frequency masking. Defaults to True.
        freq_mask_width_range (Union[int, Sequence[int]]): Range of width for frequency masks. Defaults to (0, 20).
        num_freq_mask (int): Number of frequency masks to apply. Defaults to 2.
        apply_time_mask (bool): Whether to apply time masking. Defaults to True.
        time_mask_width_range (Optional[Union[int, Sequence[int]]]): Range of width for time masks. Defaults to None.
        time_mask_width_ratio_range (Optional[Union[float, Sequence[float]]]): Range of width ratio for time masks. Defaults to None.
        num_time_mask (int): Number of time masks to apply. Defaults to 2.

    Raises:
        ValueError: If no augmentation technique is applied or if both time_mask_width_range
                    and time_mask_width_ratio_range are specified.

    Note:
        This implementation is based on the SpecAugment paper by Daniel S. Park et al.
        When using CUDA mode, time warping may not be reproducible due to
        `torch.nn.functional.interpolate`.

    Examples:
        >>> specaug = SpecAug(apply_time_warp=True, apply_freq_mask=True, apply_time_mask=True)
        >>> augmented_spec, lengths = specaug(input_spec, input_lengths)
    """

    def __init__(
        self,
        apply_time_warp: bool = True,
        time_warp_window: int = 5,
        time_warp_mode: str = "bicubic",
        apply_freq_mask: bool = True,
        freq_mask_width_range: Union[int, Sequence[int]] = (0, 20),
        num_freq_mask: int = 2,
        apply_time_mask: bool = True,
        time_mask_width_range: Optional[Union[int, Sequence[int]]] = None,
        time_mask_width_ratio_range: Optional[Union[float, Sequence[float]]] = None,
        num_time_mask: int = 2,
        replace_with_zero: bool = True,
    ):
        if not apply_time_warp and not apply_time_mask and not apply_freq_mask:
            raise ValueError(
                "Either one of time_warp, time_mask, or freq_mask should be applied"
            )
        if (
            apply_time_mask
            and (time_mask_width_range is not None)
            and (time_mask_width_ratio_range is not None)
        ):
            raise ValueError(
                'Either one of "time_mask_width_range" or '
                '"time_mask_width_ratio_range" can be used'
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
                replace_with_zero=replace_with_zero,
            )
        else:
            self.freq_mask = None

        if apply_time_mask:
            if time_mask_width_range is not None:
                self.time_mask = MaskAlongAxis(
                    dim="time",
                    mask_width_range=time_mask_width_range,
                    num_mask=num_time_mask,
                    replace_with_zero=replace_with_zero,
                )
            elif time_mask_width_ratio_range is not None:
                self.time_mask = MaskAlongAxisVariableMaxWidth(
                    dim="time",
                    mask_width_ratio_range=time_mask_width_ratio_range,
                    num_mask=num_time_mask,
                    replace_with_zero=replace_with_zero,
                )
            else:
                raise ValueError(
                    'Either one of "time_mask_width_range" or '
                    '"time_mask_width_ratio_range" should be used.'
                )
        else:
            self.time_mask = None

    def forward(self, x, x_lengths=None):
        """
                Apply SpecAugment to the input spectrogram.

        This method applies the configured augmentation techniques (time warping,
        frequency masking, and time masking) to the input spectrogram.

        Args:
            x (Tensor): Input spectrogram tensor of shape (batch_size, num_channels, num_freq, num_time).
            x_lengths (Tensor, optional): Lengths of each sequence in the batch. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]:
                - Augmented spectrogram tensor of the same shape as the input.
                - Updated lengths tensor (if x_lengths was provided, otherwise None).

        Examples:
            >>> specaug = SpecAug()
            >>> input_spec = torch.randn(32, 1, 80, 100)  # (batch_size, channels, freq, time)
            >>> input_lengths = torch.full((32,), 100)
            >>> augmented_spec, augmented_lengths = specaug.forward(input_spec, input_lengths)
        """
        if self.time_warp is not None:
            x, x_lengths = self.time_warp(x, x_lengths)
        if self.freq_mask is not None:
            x, x_lengths = self.freq_mask(x, x_lengths)
        if self.time_mask is not None:
            x, x_lengths = self.time_mask(x, x_lengths)
        return x, x_lengths
