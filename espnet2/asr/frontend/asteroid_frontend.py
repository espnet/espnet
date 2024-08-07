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
    """
        Asteroid Filterbank Frontend for audio feature extraction.

    This class implements a Sinc-convolutional-based audio feature extractor using the
    Asteroid filterbank. It provides functionality similar to using a sliding window
    frontend with a sinc preencoder.

    The frontend applies preemphasis, normalization, and frame-wise feature extraction
    using parameterized analytic filterbanks as described in Pariente et al. (2020).

    Attributes:
        sinc_filters (int): Number of filters for the sinc convolution.
        sinc_kernel_size (int): Kernel size for the sinc convolution.
        sinc_stride (int): Stride size for the first sinc-conv layer, determining
            the compression rate (Hz).
        output_dim (int): Output dimension of the feature extraction.

    Note:
        This frontend is primarily used in sentence-level classification tasks
        (e.g., speaker recognition). Its effectiveness in other applications
        has not been fully investigated.

    Example:
        >>> frontend = AsteroidFrontend(sinc_filters=256, sinc_kernel_size=251, sinc_stride=16)
        >>> input_tensor = torch.randn(32, 16000)  # (batch_size, time)
        >>> input_lengths = torch.full((32,), 16000)
        >>> output, output_lengths = frontend(input_tensor, input_lengths)
        >>> print(output.shape)  # (batch_size, time', features)

    References:
        M. Pariente, S. Cornell, A. Deleforge and E. Vincent,
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
        """
                Apply the Asteroid filterbank frontend to the input audio.

        This method processes the input audio through the Asteroid filterbank,
        applying preemphasis, normalization, and frame-wise feature extraction.

        Args:
            input (torch.Tensor): Input audio tensor of shape (B, T), where B is
                the batch size and T is the number of time steps.
            input_length (torch.Tensor): Tensor of shape (B,) containing the
                original lengths of each sequence in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Frame-wise output features of shape (B, T', D), where T' is the
                  number of frames and D is the number of features per frame.
                - Updated input lengths after processing, of shape (B,).

        Raises:
            AssertionError: If the input tensor does not have exactly 2 dimensions.

        Note:
            The forward pass temporarily disables automatic mixed precision to
            ensure consistent results.

        Example:
            >>> frontend = AsteroidFrontend(sinc_filters=256, sinc_kernel_size=251, sinc_stride=16)
            >>> input_tensor = torch.randn(32, 16000)  # (batch_size, time)
            >>> input_lengths = torch.full((32,), 16000)
            >>> output, output_lengths = frontend.forward(input_tensor, input_lengths)
            >>> print(output.shape)  # (batch_size, time', features)
            >>> print(output_lengths.shape)  # (batch_size,)
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
        """
                Return the output size of the feature dimension.

        This method returns the number of features in the output of the
        Asteroid filterbank frontend, which is equal to the number of
        sinc filters used in the convolutional layer.

        Returns:
            int: The number of features in the output, corresponding to
            the number of sinc filters.

        Example:
            >>> frontend = AsteroidFrontend(sinc_filters=256)
            >>> output_dim = frontend.output_size()
            >>> print(output_dim)  # 256
        """
        return self.sinc_filters
