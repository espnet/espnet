# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Frequency-Related Loss"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from espnet2.gan_tts.hifigan.loss import MelSpectrogramLoss


class MultiScaleMelSpectrogramLoss(torch.nn.Module):
    """
        Multi-Scale Mel Spectrogram Loss.

    This class implements a multi-scale spectrogram loss for evaluating the
    quality of generated audio waveforms compared to ground truth. It utilizes
    the Mel spectrogram representation and can be configured with various
    parameters to adjust the loss calculation across multiple scales.

    Attributes:
        alphas (List[float]): Coefficients for each scale.
        total (float): Total weight for normalizing the loss.
        normalized (bool): Indicates whether the loss is normalized.

    Args:
        fs (int): Sampling rate. Default is 22050.
        range_start (int): Power of 2 to use for the first scale. Default is 6.
        range_end (int): Power of 2 to use for the last scale. Default is 11.
        window (str): Window type for the FFT. Default is "hann".
        n_mels (int): Number of mel bins. Default is 80.
        fmin (Optional[int]): Minimum frequency for Mel. Default is 0.
        fmax (Optional[int]): Maximum frequency for Mel. Default is None.
        center (bool): Whether to use a centered window. Default is True.
        normalized (bool): Whether to use normalized spectrograms. Default is False.
        onesided (bool): Whether to use one-sided FFT. Default is True.
        log_base (Optional[float]): Log base value. Default is 10.0.
        alphas (bool): Whether to use alphas as coefficients. Default is True.

    Methods:
        forward(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            Calculates the Mel-spectrogram loss between generated and ground truth
            waveforms.

    Raises:
        AssertionError: If `range_end` is not greater than `range_start` or if
        `range_start` is less than or equal to 2.

    Examples:
        >>> loss_fn = MultiScaleMelSpectrogramLoss()
        >>> generated = torch.randn(1, 1, 1024)  # Example generated waveform
        >>> ground_truth = torch.randn(1, 1, 1024)  # Example ground truth
        >>> loss = loss_fn(generated, ground_truth)
        >>> print(loss)
    """

    def __init__(
        self,
        fs: int = 22050,
        range_start: int = 6,
        range_end: int = 11,
        window: str = "hann",
        n_mels: int = 80,
        fmin: Optional[int] = 0,
        fmax: Optional[int] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        log_base: Optional[float] = 10.0,
        alphas: bool = True,
    ):
        super().__init__()
        mel_loss = list()
        self.alphas = list()
        self.total = 0
        self.normalized = normalized
        assert range_end > range_start, "error in index"
        for i in range(range_start, range_end):
            assert range_start > 2, "range start should be more than 2 for hop_length"
            mel_loss.append(
                MelSpectrogramLoss(
                    fs=fs,
                    n_fft=int(2**i),
                    hop_length=2 ** (i - 2),
                    win_length=2**i,
                    window=window,
                    n_mels=n_mels,
                    fmin=fmin,
                    fmax=fmax,
                    center=center,
                    normalized=normalized,
                    onesided=onesided,
                    log_base=log_base,
                )
            )
            if alphas:
                self.alphas.append(np.sqrt(2**i - 1))
            else:
                self.alphas.append(1)
            self.total += self.alphas[-1] + 1

        self.mel_loss = torch.nn.ModuleList(mel_loss)

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
                Calculate Mel-spectrogram loss between generated and groundtruth waveforms.

        This method computes the multi-scale Mel-spectrogram loss by applying the
        MelSpectrogramLoss to the generated and groundtruth waveforms. The loss is
        calculated across multiple scales, combining both L1 and MSE losses, weighted
        by predefined alpha coefficients.

        Args:
            y_hat (torch.Tensor): Generated waveform tensor of shape (B, 1, T), where
                B is the batch size and T is the number of time steps.
            y (torch.Tensor): Groundtruth waveform tensor of shape (B, 1, T).

        Returns:
            torch.Tensor: A scalar tensor representing the calculated Mel-spectrogram
                loss value.

        Examples:
            >>> loss_fn = MultiScaleMelSpectrogramLoss()
            >>> generated_waveform = torch.randn(2, 1, 1024)  # Example shape
            >>> groundtruth_waveform = torch.randn(2, 1, 1024)  # Example shape
            >>> loss = loss_fn(generated_waveform, groundtruth_waveform)
            >>> print(loss)
        """
        loss = 0.0
        for i in range(len(self.alphas)):
            l1 = self.mel_loss[i](y_hat, y)
            l2 = self.mel_loss[i](y_hat, y, use_mse=True)
            loss += l1 + self.alphas[i] * l2
        loss = loss / self.total
        return loss
