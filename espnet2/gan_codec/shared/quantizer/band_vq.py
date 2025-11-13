# Copyright 2025 Haoran Wang
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import math
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from espnet2.gan_codec.shared.quantizer.modules.core_vq import BandVectorQuantization
from espnet2.gan_codec.shared.quantizer.modules.simvq import SimVQ


class BandVQ(nn.Module):

    def __init__(
        self,
        num_bands: int = 3,
        dimension: int = 512,
        codebook_dim: int = 512,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        quantizer_dropout: bool = False,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.dimension = dimension
        self.codebook_dim = codebook_dim
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.quantizer_dropout = quantizer_dropout
        self.vq = BandVectorQuantization(
            num_bands=self.num_bands,
            dim=self.dimension,
            codebook_dim=self.codebook_dim,
            codebook_size=self.bins,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            quantizer_dropout=self.quantizer_dropout,
        )

    def forward(self, x: torch.Tensor, bandwidth: Optional[float] = None):
        quantized, codes, commit_loss = self.vq(x)
        return (quantized, codes, torch.mean(commit_loss))

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        codes = self.vq.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.vq.decode(codes)
        return quantized


class BandSimVQ(nn.Module):

    def __init__(
        self,
        num_bands: int,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        channel_first: bool = True,
        rotation_trick: bool = False,
    ):
        super().__init__()

        self.num_bands = num_bands
        self.dim = dim

        self.layers = nn.ModuleList(
            [
                SimVQ(
                    dim=dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    rotation_trick=rotation_trick,
                    channel_first=True,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Band Simple VQ.

        Args:
            x: Input tensor of shape [B, bands, D, T].

        Returns:
            quantized_per_band: Quantized output [B, bands, D, T].
            all_indices: Indices [B, bands, T].
            total_loss: Total VQ loss (scalar).
        """
        B, bands, D, T = x.shape
        assert bands == self.num_bands

        quantized_per_band = []
        all_indices = []
        all_simvq_losses = []

        for i in range(bands):
            inp = x[:, i, :, :]
            quantized, indices, simvq_loss = self.layers[i](inp)
            quantized_per_band.append(quantized)
            all_indices.append(indices)
            all_simvq_losses.append(simvq_loss)

        quantized_per_band = torch.stack(quantized_per_band, dim=1)
        all_indices = torch.stack(all_indices, dim=1)

        total_loss = torch.stack(all_simvq_losses).mean()

        return quantized_per_band, all_indices, total_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to indices.

        Args:
            x: Input tensor [B, bands, D, T].

        Returns:
            indices: Indices tensor [B, bands, T].
        """
        _, indices, _ = self.forward(x)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode indices to reconstructed signal.

        Args:
            indices: Indices tensor [B, bands, T].

        Returns:
            reconstructed: Reconstructed signal [B, bands, D, T].
        """
        B, bands = indices.shape[:2]
        assert bands == self.num_bands

        reconstructed = []
        for i in range(bands):
            codes = self.layers[i].indices_to_codes(indices[:, i, :])
            reconstructed.append(codes)

        return torch.stack(reconstructed, dim=1)
