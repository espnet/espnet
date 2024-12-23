# Copyright 2024 Jiatong Shi
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted from https://github.com/facebookresearch/encodec

# Original license as follows:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

"""Residual vector quantizer implementation."""
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from espnet2.gan_codec.shared.quantizer.modules.core_vq import (
    ResidualVectorQuantization,
)


@dataclass
class QuantizedResult:
    """
        Represents the result of quantization from a residual vector quantizer.

    Attributes:
        quantized (torch.Tensor): The quantized representation of the input tensor.
        codes (torch.Tensor): The codes representing the quantized values.
        bandwidth (torch.Tensor): Bandwidth in kb/s used, per batch item.
        penalty (Optional[torch.Tensor]): Optional penalty term for the loss, if
            applicable.

    Examples:
        >>> result = QuantizedResult(
        ...     quantized=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        ...     codes=torch.tensor([[1, 2], [3, 4]]),
        ...     bandwidth=torch.tensor([128.0]),
        ...     penalty=torch.tensor([0.05])
        ... )
        >>> print(result.quantized)
        tensor([[0.1, 0.2],
                [0.3, 0.4]])
        >>> print(result.codes)
        tensor([[1, 2],
                [3, 4]])
        >>> print(result.bandwidth)
        tensor([128.])
        >>> print(result.penalty)
        tensor([0.05])
    """

    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: Optional[torch.Tensor] = None


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer.

    This class implements a residual vector quantization mechanism which is
    designed to quantize input tensors while minimizing information loss. It
    utilizes multiple quantizers to achieve better performance and supports
    initialization via k-means.

    Attributes:
        n_q (int): Number of residual vector quantizers used.
        dimension (int): Dimension of the codebooks.
        codebook_dim (int): Dimension of the codebook vectors.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
        quantizer_dropout (bool): Flag to indicate whether to apply dropout.

    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
            Replace any codes that have an exponential moving average cluster
            size less than the specified threshold with a randomly selected
            vector from the current batch.

    Examples:
        >>> rvq = ResidualVectorQuantizer(dimension=256, n_q=8, bins=1024)
        >>> input_tensor = torch.randn(1, 256)
        >>> sample_rate = 16000
        >>> result = rvq(input_tensor, sample_rate)
        >>> quantized_output = result.quantized
        >>> codes = result.codes
    """

    def __init__(
        self,
        dimension: int = 256,
        codebook_dim: int = 512,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        quantizer_dropout: bool = False,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.codebook_dim = codebook_dim
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.quantizer_dropout = quantizer_dropout
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_dim=self.codebook_dim,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            quantizer_dropout=self.quantizer_dropout,
        )

    def forward(
        self, x: torch.Tensor, sample_rate: int, bandwidth: Optional[float] = None
    ) -> QuantizedResult:
        """
        Residual vector quantization on the given input tensor.

        This method performs residual vector quantization on the input tensor
        `x` and returns a `QuantizedResult` containing the quantized
        representation, associated bandwidth, and any penalty term for the
        loss. The number of quantizers used is determined based on the
        specified bandwidth and sample rate.

        Args:
            x (torch.Tensor): Input tensor.
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (Optional[float]): Target bandwidth for quantization.

        Returns:
            QuantizedResult:
                A dataclass containing the quantized representation, codes,
                bandwidth, and an optional penalty term. The attributes are:
                - quantized (torch.Tensor): The quantized representation.
                - codes (torch.Tensor): The indices of the quantized codes.
                - bandwidth (torch.Tensor): The bandwidth in kb/s used per
                  batch item.
                - penalty (Optional[torch.Tensor]): An optional penalty term
                  for the loss, if applicable.

        Examples:
            >>> model = ResidualVectorQuantizer()
            >>> input_tensor = torch.randn(1, 256)
            >>> sample_rate = 16000
            >>> result = model.forward(input_tensor, sample_rate)
            >>> print(result.quantized.shape)  # Output shape of quantized tensor
            >>> print(result.bandwidth)          # Output bandwidth

        Note:
            If `quantizer_dropout` is enabled, the method may also return
            an additional quantization loss term.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)

        if not self.quantizer_dropout:
            quantized, codes, commit_loss = self.vq(x, n_q=n_q)
            bw = torch.tensor(n_q * bw_per_q).to(x)
            return quantized, codes, bw, torch.mean(commit_loss)
        else:
            quantized, codes, commit_loss, quantization_loss = self.vq(x, n_q=n_q)
            bw = torch.tensor(n_q * bw_per_q).to(x)
            return (
                quantized,
                codes,
                bw,
                torch.mean(commit_loss),
                torch.mean(quantization_loss),
            )

    def get_num_quantizers_for_bandwidth(
        self, sample_rate: int, bandwidth: Optional[float] = None
    ) -> int:
        """
            Return the number of quantizers (n_q) based on specified target bandwidth.

        This method calculates the number of residual vector quantizers that can be
        utilized given a target bandwidth. It ensures that the number of quantizers
        does not exceed the available bandwidth divided by the bandwidth per quantizer.

        Args:
            sample_rate (int): The sample rate of the input data.
            bandwidth (Optional[float]): The target bandwidth in kb/s. If None or
                less than or equal to zero, the maximum number of quantizers (n_q)
                is returned.

        Returns:
            int: The number of quantizers to use based on the specified bandwidth.

        Examples:
            >>> rvq = ResidualVectorQuantizer()
            >>> rvq.get_num_quantizers_for_bandwidth(sample_rate=44100, bandwidth=64)
            2

            >>> rvq.get_num_quantizers_for_bandwidth(sample_rate=44100, bandwidth=None)
            8

        Note:
            The bandwidth must be a positive value to calculate the number of
            quantizers. If the bandwidth is invalid, the method defaults to the
            maximum number of quantizers defined during initialization.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.0:
            n_q = int(max(1, math.floor(bandwidth / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, sample_rate: int):
        """
        Return bandwidth per quantizer for a given input sample rate.

        This method calculates the bandwidth required for each quantizer based
        on the input sample rate. The bandwidth is computed using the formula:
        bandwidth = log2(bins) * sample_rate / 1000, where 'bins' refers to
        the codebook size.

        Args:
            sample_rate (int): The sample rate of the input tensor.

        Returns:
            float: The bandwidth per quantizer in kilobits per second (kb/s).

        Examples:
            >>> rvq = ResidualVectorQuantizer()
            >>> bw = rvq.get_bandwidth_per_quantizer(sample_rate=16000)
            >>> print(bw)
            128.0  # This value may vary depending on the bins and sample rate.
        """
        return math.log2(self.bins) * sample_rate / 1000

    def encode(
        self,
        x: torch.Tensor,
        sample_rate: int,
        bandwidth: Optional[float] = None,
        st: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode a given input tensor with the specified sample rate at
        the given bandwidth.

        The RVQ encode method sets the appropriate number of quantizers to
        use and returns indices for each quantizer.

        Args:
            x (torch.Tensor): Input tensor to be encoded.
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (Optional[float]): Target bandwidth. If specified,
                the number of quantizers will be adjusted based on this
                value.
            st (Optional[int]): Starting index for encoding. Defaults to 0.

        Returns:
            torch.Tensor: Indices for each quantizer after encoding.

        Examples:
            >>> rvq = ResidualVectorQuantizer()
            >>> input_tensor = torch.randn(1, 256)  # Example input
            >>> sample_rate = 16000
            >>> bandwidth = 128
            >>> encoded_indices = rvq.encode(input_tensor, sample_rate, bandwidth)

        Note:
            Ensure that the input tensor has the correct dimensions for
            encoding.
        """
        n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        st = st or 0
        codes = self.vq.encode(x, n_q=n_q, st=st)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
            Decode the given codes to the quantized representation.

        This method takes the quantization codes produced by the encoding process
        and converts them back into the quantized tensor representation.

        Args:
            codes (torch.Tensor): A tensor containing the quantization codes to be
                decoded.

        Returns:
            torch.Tensor: The decoded quantized representation corresponding to
                the provided codes.

        Examples:
            >>> rvq = ResidualVectorQuantizer()
            >>> codes = torch.tensor([[1, 2], [3, 4]])
            >>> decoded_output = rvq.decode(codes)
            >>> print(decoded_output.shape)  # Output will depend on the quantizer setup

        Note:
            The input `codes` should be properly formatted as per the quantizer's
            requirements for decoding to succeed.
        """
        quantized = self.vq.decode(codes)
        return quantized
