import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(
        self,
        orig_dim: int,
        down_dim: int,
        layer_norm: str = None,
        activation_fn: str = "gelu",
    ) -> None:
        super().__init__()

        self.down_projection = nn.Linear(orig_dim, down_dim)
        self.up_projection = nn.Linear(down_dim, orig_dim)
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.xavier_uniform_(self.up_projection.weight)
        self.activation = nn.GELU()

        self.layer_norm_opt = layer_norm
        if layer_norm is not None:
            self.layer_norm = nn.LayerNorm(orig_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_norm_opt == "first":
            x = self.layer_norm(x)

        x = self.down_projection(x)
        x = self.activation(x)
        x = self.up_projection(x)

        if self.layer_norm_opt == "last":
            x = self.layer_norm(x)

        return x
