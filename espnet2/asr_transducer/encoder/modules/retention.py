"""Multi-scale retention module.

Based/modified from https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiscale_retention.py

"""  # noqa

import math
from typing import Optional, Tuple

import torch


class MultiScaleRetention(torch.nn.Module):
    """MultiScaleRetention module definition.

    Args:
        num_heads: Number of attention heads.
        size: Hidden size.
        activation: Activation module.
        num_blocks: Number of blocks in the architecture for init scaling.
        value_factor: Size factor for the value.
        dropout_rate: Dropout rate for the retention module.

    """

    def __init__(
        self,
        size: int,
        num_heads: int,
        activation: torch.nn.Module,
        num_blocks: int,
        value_factor: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a MultiScaleRetention object."""
        super().__init__()

        head_size = (size * value_factor) // num_heads
        qk_size = size // num_heads
        scaling = qk_size**-0.5

        self.proj_query = torch.nn.Linear(size, size)
        self.proj_key = torch.nn.Linear(size, size)

        self.proj_value = torch.nn.Linear(size, size * value_factor)
        self.proj_g = torch.nn.Linear(size, size * value_factor)

        self.proj_output = torch.nn.Linear(size * value_factor, size)

        self.activation = activation
        self.norm = torch.nn.LayerNorm(head_size, elementwise_affine=False)

        self.hidden_dropout = torch.nn.Dropout(p=dropout_rate)

        self.size = size
        self.head_size = head_size
        self.qk_size = qk_size

        self.scaling = scaling
        self.value_factor = value_factor

        self.num_heads = num_heads

        self.reset_parameters(num_blocks)

    def reset_parameters(self, num_blocks: int) -> None:
        """Reset module parameters.

        Args:
            num_blocks: Number of blocks in the architecture.

        """
        init_scale = math.sqrt(math.pow(8.0 * num_blocks, 0.25))
        gain = 2**-2.5

        torch.nn.init.xavier_uniform_(self.proj_query.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.proj_key.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.proj_value.weight, gain=gain)

        torch.nn.init.xavier_uniform_(self.proj_g.weight, gain=gain)

        torch.nn.init.xavier_uniform_(self.proj_output.weight)
        torch.nn.init.constant_(self.proj_output.bias, 0.0)

        self.proj_value.weight.data.mul_(init_scale)
        self.proj_value.bias.data.mul_(init_scale)

        self.proj_output.weight.data.mul_(init_scale)
        self.proj_output.bias.data.mul_(init_scale)

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate every two elements in the last dimension.

        Args:
            x: Input sequence. (B, num_heads, length, D_qk)

        Returns:
            x: Rotated output sequence. (B, num_heads, length, D_qk)

        """
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x = torch.stack((-x2, x1), dim=-1)

        return x.flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        pos: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform multi-scale retention.

        Args:
            x: MultiScaleRetention input sequences. (B, length, D_enc)
            pos: Positional embeddings.
                   ((length, D_angle), (length, D_angle), (num_heads, length, length))
                   or ((D_angle), (D_angle), (D_angle))
            state: MultiScaleRetention state. {(??), (??)} or {}

        Returns:
            x_out: MultiScaleRetention output sequences. (B, T, D_enc)
            state: MultiScaleRetention state. {(??), (??)} or {}

        """
        batch, length, size = x.size()

        (cos, sin), inner_mask = pos

        query = self.proj_query(x)
        key = self.proj_key(x) * self.scaling
        value = self.proj_value(x)

        g = self.proj_g(x)

        query = query.view(batch, length, self.num_heads, self.qk_size).transpose(1, 2)
        key = key.view(batch, length, self.num_heads, self.qk_size).transpose(1, 2)

        rotated_query = (query * cos) + (self.rotate_every_two(query) * sin)
        rotated_key = (key * cos) + (self.rotate_every_two(key) * sin)

        if state is not None:
            x, state = self.chunk_forward(
                rotated_query, rotated_key, value, inner_mask, state=state
            )
        else:
            x = self.parallel_forward(rotated_query, rotated_key, value, inner_mask)

        x = self.norm(x).reshape(batch, length, self.head_size * self.num_heads)
        x = self.proj_output(self.activation(g) * x)

        return x, state

    def parallel_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perform parallel retention.

        Args:
            query: Rotated query tensor. (B, length, num_heads, D_qk)
            key: Rotated key tensor. (B, length, num_heads, D_qk)
            value: Value tensor. (B, length, num_heads, value_factor * size)
            mask: Decaying mask. (num_heads)

        Returns:
            x: Parallel retention output. (B, length, num_heads, size)

        """
        batch, length, size = value.size()

        value = value.view(batch, length, self.num_heads, self.head_size).transpose(
            1, 2
        )

        query_key = (query @ key.transpose(-1, -2)) * mask
        query_key /= query_key.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        query_key = self.hidden_dropout(query_key)

        x = torch.matmul(query_key, value).transpose(1, 2)

        return x

    def chunk_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform chunk-wise recurrent retention.

        Args:
            query: Rotated query tensor. (B, L, num_heads, D_qk)
            key: Rotated key tensor. (B, L, num_heads, D_qk)
            value: Value tensor. (B, L, size)
            mask: Inner mask. (num_heads)
            state: MultiScaleRetention state. {(??), (??)} or {}

        Returns:
            x: Recurrent retention output. (B, length, num_heads, size)
            state: MultiScaleRetention state. {(??), (??)} or {}

        """
        raise NotImplementedError
