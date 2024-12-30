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
    AsteroidFrontend class for audio feature extraction using Sinc-convolution.

    This class implements a Sinc-convolutional-based audio feature extractor
    designed for tasks such as sentence-level classification. It utilizes a
    parameterized analytic filterbank layer to process raw audio input data
    and extract meaningful features.

    The functionality of this class can also be achieved by combining a
    sliding window frontend with a Sinc preencoder.

    Attributes:
        sinc_filters (int): Number of filters for Sinc convolution.
        sinc_kernel_size (int): Kernel size for Sinc convolution.
        sinc_stride (int): Stride size for the first Sinc convolution layer.
        preemph_coef (float): Coefficient for preemphasis applied to the input.
        log_term (float): A small constant added to prevent log of zero.

    Args:
        sinc_filters (int): The number of Sinc filters. Default is 256.
        sinc_kernel_size (int): The kernel size for Sinc convolution. Default is 251.
        sinc_stride (int): The stride size for the Sinc convolution layer. Default is 16.
        preemph_coef (float): The coefficient for preemphasis. Default is 0.97.
        log_term (float): The log term to prevent infinity. Default is 1e-6.

    Methods:
        forward(input: torch.Tensor, input_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Applies the Asteroid filterbank frontend to the input tensor and returns
            the frame-wise output along with the adjusted input lengths.

        output_size() -> int:
            Returns the output length of the feature dimension.

    Examples:
        >>> frontend = AsteroidFrontend(sinc_filters=128, sinc_kernel_size=251)
        >>> input_tensor = torch.randn(10, 16000)  # (B, T)
        >>> input_length = torch.tensor([16000] * 10)  # (B,)
        >>> output, output_length = frontend(input_tensor, input_length)

    Note:
        This class is primarily used for tasks related to speech separation
        and classification. Other applications are not thoroughly examined.

    Raises:
        AssertionError: If the input tensor does not have 2 dimensions.
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
        Apply the Asteroid filterbank frontend to the input audio data.

        This method processes the input audio tensor using the Asteroid filterbank
        to extract frame-wise features suitable for downstream tasks. It includes
        preemphasis, normalization, and feature extraction through a Sinc-based
        convolutional layer.

        Args:
            input: A tensor of shape (B, T) representing the audio input, where B
                is the batch size and T is the length of the audio sequence.
            input_length: A tensor of shape (B,) containing the lengths of each
                audio sequence in the batch.

        Returns:
            A tuple containing:
                - Tensor: Frame-wise output of shape (B, T', D), where T' is the
                  number of frames after processing and D is the output feature
                  dimension.
                - Tensor: Updated input lengths after processing, of shape (B,).

        Raises:
            AssertionError: If the input tensor does not have 2 dimensions.

        Examples:
            >>> frontend = AsteroidFrontend()
            >>> audio_input = torch.randn(4, 16000)  # Batch of 4, 1 second audio
            >>> input_length = torch.tensor([16000, 16000, 16000, 16000])
            >>> output, new_lengths = frontend(audio_input, input_length)
            >>> output.shape
            torch.Size([4, T', 256])  # Example output shape
            >>> new_lengths
            tensor([T', T', T', T'])  # Updated lengths after processing

        Note:
            This function is primarily used in sentence-level classification tasks
            such as speaker recognition. Other use cases may not be fully explored.
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

        This method provides the size of the output features generated by the
        Asteroid filterbank frontend. The output size corresponds to the
        number of sinc filters specified during the initialization of the
        AsteroidFrontend class.

        Returns:
            int: The number of sinc filters used in the feature extraction.

        Examples:
            >>> frontend = AsteroidFrontend(sinc_filters=256)
            >>> output_size = frontend.output_size()
            >>> print(output_size)
            256
        """
        return self.sinc_filters
