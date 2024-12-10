"""SpecAugment module."""

from typing import Optional, Sequence, Union

from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.mask_along_axis import MaskAlongAxis, MaskAlongAxisVariableMaxWidth
from espnet2.layers.time_warp import TimeWarp


class SpecAug(AbsSpecAug):
    """
    SpecAugment module for applying various data augmentation techniques on
    spectrograms for Automatic Speech Recognition (ASR).

    This class implements the SpecAugment method, which introduces several
    augmentation techniques, including time warping, frequency masking, and
    time masking. It is designed to improve the robustness of ASR systems by
    enhancing the training dataset through these augmentations.

    Reference:
        Daniel S. Park et al.
        "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
        Recognition"

    Warning:
        When using CUDA mode, time_warp does not guarantee reproducibility due
        to `torch.nn.functional.interpolate`.

    Attributes:
        apply_time_warp (bool): Flag to apply time warping.
        apply_freq_mask (bool): Flag to apply frequency masking.
        apply_time_mask (bool): Flag to apply time masking.
        time_warp (TimeWarp): Instance of the TimeWarp class for time
            warping.
        freq_mask (MaskAlongAxis): Instance of the MaskAlongAxis class for
            frequency masking.
        time_mask (Union[MaskAlongAxis, MaskAlongAxisVariableMaxWidth]): Instance
            of the MaskAlongAxis or MaskAlongAxisVariableMaxWidth class for
            time masking.

    Args:
        apply_time_warp (bool): If True, apply time warping. Defaults to True.
        time_warp_window (int): Window size for time warping. Defaults to 5.
        time_warp_mode (str): Interpolation mode for time warping. Defaults to
            "bicubic".
        apply_freq_mask (bool): If True, apply frequency masking. Defaults to
            True.
        freq_mask_width_range (Union[int, Sequence[int]]): Range of width for
            frequency masking. Defaults to (0, 20).
        num_freq_mask (int): Number of frequency masks to apply. Defaults to 2.
        apply_time_mask (bool): If True, apply time masking. Defaults to True.
        time_mask_width_range (Optional[Union[int, Sequence[int]]]): Range of
            width for time masking. Defaults to None.
        time_mask_width_ratio_range (Optional[Union[float, Sequence[float]]]):
            Ratio range for time masking width. Defaults to None.
        num_time_mask (int): Number of time masks to apply. Defaults to 2.
        replace_with_zero (bool): If True, replace masked values with zero.
            Defaults to True.

    Raises:
        ValueError: If none of the augmentation methods are applied, or if both
            `time_mask_width_range` and `time_mask_width_ratio_range` are set.

    Examples:
        # Create a SpecAug instance with default parameters
        spec_aug = SpecAug()

        # Apply augmentation on a batch of audio features
        augmented_features, augmented_lengths = spec_aug.forward(features, lengths)

    Note:
        The augmentations are applied in the following order: time warping,
        frequency masking, and then time masking.
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
        Apply the SpecAugment transformations to the input audio tensor.

        This method applies a series of augmentations, including time warping,
        frequency masking, and time masking, to the input audio tensor `x`.
        Each augmentation is applied sequentially based on the specified
        parameters during the initialization of the SpecAug class.

        Args:
            x (torch.Tensor): The input audio tensor with shape
                (batch_size, num_channels, time_steps, freq_bins).
            x_lengths (Optional[torch.Tensor]): An optional tensor containing
                the lengths of the input sequences. It has shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                A tuple containing the augmented audio tensor and the
                updated lengths tensor. If `x_lengths` is not provided,
                the second element of the tuple will be None.

        Raises:
            ValueError: If the input tensor `x` is not of the expected shape.

        Examples:
            >>> import torch
            >>> specaug = SpecAug()
            >>> audio_tensor = torch.rand(4, 1, 16000, 80)  # Example tensor
            >>> lengths = torch.tensor([16000, 16000, 16000, 16000])  # Example lengths
            >>> augmented_tensor, augmented_lengths = specaug.forward(audio_tensor, lengths)

        Note:
            The augmentations are performed in the following order:
            1. Time Warping
            2. Frequency Masking
            3. Time Masking

        Warning:
            When using CUDA mode, the time warping may not be reproducible due to
            `torch.nn.functional.interpolate`.
        """
        if self.time_warp is not None:
            x, x_lengths = self.time_warp(x, x_lengths)
        if self.freq_mask is not None:
            x, x_lengths = self.freq_mask(x, x_lengths)
        if self.time_mask is not None:
            x, x_lengths = self.time_mask(x, x_lengths)
        return x, x_lengths
