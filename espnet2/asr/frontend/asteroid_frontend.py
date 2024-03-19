#!/usr/bin/env python3
#  2020, Technische UniversitÃ¤t MÃ¼nchen;  Ludwig KÃ¼rzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sliding Window for raw audio input data."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, ParamSincFB
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class AsteroidFrontend(AbsFrontend):
    """Asteroid Filterbank Frontend.

    Provides a Sinc-convolutional-based audio feature extractor. The same
    function can be achieved by using `sliding_winodw frontend +
    sinc preencoder`.

    NOTE(jiatong): this function is used in sentence-level classification
    tasks (e.g., spk). Other usages are not fully investigated.

    NOTE(jeeweon): this function implements the parameterized analytic
    filterbank layer in M. Pariente, S. Cornell, A. Deleforge and E. Vincent,
    "Filterbank design for end-to-end speech separation," in Proc. ICASSP, 2020
    """

    @typechecked
    def __init__(
        self,
        sinc_filters: int = 256,
        sinc_kernel_size: int = 251,
        sinc_stride: int = 16,
        preemph_coef: float = 0.97,
        log_term: float = 1e-6,
    ):
        """Initialize.

        Args:
            sinc_filters: the filter numbers for sinc.
            sinc_kernel_size: the kernel size for sinc.
            sinc_stride: the sincstride size of the first sinc-conv layer
                where it decides the compression rate (Hz).
            preemph_coef: the coeifficient for preempahsis.
            log_term: the log term to prevent infinity.
        """
        super().__init__()

        # kernel for preemphasis
        # In pytorch, the convolution operation uses cross-correlation,
        # so the filter is flipped
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-preemph_coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

        self.norm = nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        self.sinc_filters = sinc_filters
        self.conv = Encoder(
            ParamSincFB(sinc_filters, sinc_kernel_size, stride=sinc_stride)
        )
        self.log_term = log_term
        self.sinc_kernel_size = sinc_kernel_size
        self.sinc_stride = sinc_stride
        self.output_dim = sinc_filters

    def forward(
        self, input: torch.Tensor, input_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the Asteroid filterbank frontend to the input.

        Args:
            input: Input (B, T).
            input_length: Input length (B,).

        Returns:
            Tensor: Frame-wise output (B, T', D).
        """
        # input check
        assert (
            len(input.size()) == 2
        ), "The number of dimensions of input tensor must be 2!"

        with torch.cuda.amp.autocast(enabled=False):
            # reflect padding to match lengths of in/out
            x = input.unsqueeze(1)
            x = F.pad(x, (1, 0), "reflect")

            # apply preemphasis
            x = F.conv1d(x, self.flipped_filter)

            # apply norm
            x = self.norm(x)

            # apply frame feature extraction
            x = torch.log(torch.abs(self.conv(x)) + self.log_term)

        input_length = (input_length - self.sinc_kernel_size) // self.sinc_stride + 1
        x = x - torch.mean(x, dim=-1, keepdim=True)

        return x.permute(0, 2, 1), input_length

    def output_size(self) -> int:
        """Return output length of feature dimension D."""
        return self.sinc_filters
