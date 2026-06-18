"""BestRQ tokenizer.
https://arxiv.org/abs/2202.01855
"""

import torch
from torch import nn


class RandomProjectionQuantizer(nn.Module):
    def __init__(self, dim, codebook_size, codebook_dim, norm=True, seed=45):
        super().__init__()
        # Random projection layer
        self.random_projection = nn.Linear(dim, codebook_dim, bias=False)
        # NOTE(shikhar): xavier_normal has less skew
        nn.init.xavier_normal_(self.random_projection.weight)

        self.maybe_norm = (
            nn.LayerNorm(codebook_dim, elementwise_affine=False)
            if norm
            else nn.Identity()
        )
        # Codebook initialized with the local generator
        self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim))
        self.random_projection.weight.requires_grad = False
        self.codebook.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform random projection and quantization.

        Args:
            x (torch.Tensor): Input tensor with shape `(B, L, D)`.
        Returns:
            torch.Tensor: Output tensor of shape `(B, L)` containing quantized indices.
        """
        x = self.random_projection(x)  # Ax
        x = self.maybe_norm(x)  # norml2(Ax)
        # Squared L2 distance to each codebook entry, computed via
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x . c  to avoid materializing
        # the (B, L, K, D) broadcast (OOMs on CI with default K=1024, D=256).
        # argmin over distance == argmin over distance^2.
        x_sq = (x * x).sum(dim=-1, keepdim=True)  # (B, L, 1)
        codebook_sq = (self.codebook * self.codebook).sum(dim=-1)  # (K,)
        distance_sq = x_sq + codebook_sq - 2 * x @ self.codebook.t()  # (B, L, K)
        codes = torch.argmin(distance_sq, dim=-1)
        return codes
