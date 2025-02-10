"""BestRQ tokenizer.
https://arxiv.org/abs/2202.01855
"""

import torch
from torch import nn
from torch.linalg import vector_norm


class RandomProjectionQuantizer(nn.Module):
    def __init__(self, dim, codebook_size, codebook_dim, norm=True, seed=42):
        super().__init__()
        # Seed locally for reproducibility
        original_rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        # Random projection layer
        self.random_projection = nn.Linear(dim, codebook_dim, bias=False)
        # NOTE(shikhar): change this to xavier_normal (better skew in codebook)
        nn.init.xavier_uniform_(self.random_projection.weight)

        self.maybe_norm = (
            nn.LayerNorm(codebook_dim, elementwise_affine=False)
            if norm
            else nn.Identity()
        )
        # Codebook initialized with the local generator
        self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim))
        torch.set_rng_state(original_rng_state)
        self.random_projection.weight.requires_grad = False
        self.codebook.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform random projection and quantization.
        Args:
            x (torch.Tensor): Input tensor with shape `(B, L, D)`.
        Returns:
            torch.Tensor: Output tensor of shape `(B, L)` containing quantized indices.
        """
        x = self.random_projection(x)  # Ax
        x = self.maybe_norm(x)  # norml2(Ax)
        distance = vector_norm(
            x.unsqueeze(-2) - self.codebook, dim=-1
        )  # B,L,codebook_size
        codes = torch.argmin(distance, dim=-1)
        return codes
