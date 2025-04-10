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
from dataclasses import dataclass, field  # noqa
from typing import Optional

import torch
from torch import nn

from espnet2.gan_codec.shared.quantizer.modules.core_vq import (
    ResidualVectorQuantization,
)


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: Optional[torch.Tensor] = None


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.

    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
            Replace any codes that have an exponential moving average cluster
            size less than the specified threshold with
            randomly selected vector from the current batch.
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
        """Residual vector quantization on the given input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
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
        """Return n_q based on specified target bandwidth."""
        bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.0:
            n_q = int(max(1, math.floor(bandwidth / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, sample_rate: int):
        """Return bandwidth per quantizer for a given input sample rate."""
        return math.log2(self.bins) * sample_rate / 1000

    def encode(
        self,
        x: torch.Tensor,
        sample_rate: int,
        bandwidth: Optional[float] = None,
        st: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at

        the given bandwidth. The RVQ encode method sets the appropriate
        number of quantizer to use and returns indices for each quantizer.
        """
        n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        st = st or 0
        codes = self.vq.encode(x, n_q=n_q, st=st)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        quantized = self.vq.decode(codes)
        return quantized
