#!/usr/bin/env python3
#  2023, Jee-weon Jung, CMU
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Torchaudio MFCC"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class MelSpectrogramTorch(AbsFrontend):
    """
    Mel-Spectrogram using Torchaudio Implementation.
    """

    def __init__(
        self,
        preemp: bool = True,
        n_fft: int = 512,
        log: bool = False,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: int = 20,
        f_max: int = 7600,
        n_mels: int = 80,
        window_fn: str = "hamming",
        mel_scale: str = "htk",
        normalize: str = None,
    ):
        assert check_argument_types()
        super().__init__()

        self.log = log
        self.n_mels = n_mels
        self.preemp = preemp
        self.normalize = normalize
        if window_fn == "hann":
            self.window_fn = torch.hann_window
        elif window_fn == "hamming":
            self.window_fn = torch.hamming_window

        if preemp:
            self.register_buffer(
                "flipped_filter",
                torch.FloatTensor([-0.97, 1.0]).unsqueeze(0).unsqueeze(0),
            )

        self.transform = ta.transforms.MelSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=self.window_fn,
            mel_scale=mel_scale,
        )

    def forward(
        self, input: torch.Tensor, input_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input check
        assert (
            len(input.size()) == 2
        ), "The number of dimensions of input tensor must be 2!"

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if self.preemp:
                    # reflect padding to match lengths of in/out
                    x = input.unsqueeze(1)
                    x = F.pad(x, (1, 0), "reflect")

                    # apply preemphasis
                    x = F.conv1d(x, self.flipped_filter).squeeze(1)
                else:
                    x = input

                # apply frame feature extraction
                x = self.transform(x)

                if self.log:
                    x = torch.log(x + 1e-6)
                if self.normalize is not None:
                    if self.normalize == "mn":
                        x = x - torch.mean(x, dim=-1, keepdim=True)
                    else:
                        raise NotImplementedError(
                            f"got {self.normalize}, not implemented"
                        )

        input_length = torch.Tensor([x.size(-1)]).repeat(x.size(0))

        return x.permute(0, 2, 1), input_length

    def output_size(self) -> int:
        """Return output length of feature dimension D."""
        return self.n_mels
