# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Frequency-Related Loss"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa

from espnet2.gan_tts.hifigan.loss import MelSpectrogramLoss


class MultiScaleMelSpectrogramLoss(torch.nn.Module):
    """Multi-Scale spectrogram loss.

    Args:
        fs (int): Sampling rate.
        range_start (int): Power of 2 to use for the first scale.
        range_stop (int): Power of 2 to use for the last scale.
        window (str): Window type.
        n_mels (int): Number of mel bins.
        fmin (Optional[int]): Minimum frequency for Mel.
        fmax (Optional[int]): Maximum frequency for Mel.
        center (bool): Whether to use center window.
        normalized (bool): Whether to use normalized one.
        onesided (bool): Whether to use oneseded one.
        log_base (Optional[float]): Log base value.
        alphas (bool): Whether to use alphas as coefficients or not..
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
        """Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated waveform tensor (B, 1, T).
            y (Tensor): Groundtruth waveform tensor (B, 1, T).


        Returns:
            Tensor: Mel-spectrogram loss value.
        """
        loss = 0.0
        for i in range(len(self.alphas)):
            l1 = self.mel_loss[i](y_hat, y)
            l2 = self.mel_loss[i](y_hat, y, use_mse=True)
            loss += l1 + self.alphas[i] * l2
        loss = loss / self.total
        return loss
