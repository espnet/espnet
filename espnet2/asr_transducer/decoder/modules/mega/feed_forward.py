"""Normalized position-wise feed-forward module for MEGA block."""

import torch


class NormalizedPositionwiseFeedForward(torch.nn.Module):
    """NormalizedPositionFeedForward module definition.

    Args:
        size: Input/Output size.
        hidden_size: Hidden size.
        normalization: Normalization module.
        activation: Activation function.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        size: int,
        hidden_size: int,
        normalization: torch.nn.Module = torch.nn.LayerNorm,
        activation: torch.nn.Module = torch.nn.ReLU,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct an NormalizedPositionwiseFeedForward object."""
        super().__init__()
        self.linear1 = torch.nn.Linear(size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, size)

        self.normalization = normalization
        self.activation = activation

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(p=dropout_rate)

        self.reset_parameters()

    def reset_parameters(self, val: float = 0.0, std: float = 0.02) -> None:
        """Reset module parameters.

        Args:
            val: Initialization value.
            std: Standard deviation.

        """
        torch.nn.init.normal_(self.linear1.weight, mean=val, std=std)
        torch.nn.init.constant_(self.linear1.bias, val)

        torch.nn.init.normal_(self.linear2.weight, mean=val, std=std)
        torch.nn.init.constant_(self.linear2.bias, val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute feed-forward module.

        Args:
            x: NormalizedPositionwiseFeedForward input sequences. (B, L, size)

        Returns:
            x: NormalizedPositionwiseFeedForward output sequences. (B, L, size)

        """
        residual = x

        x = self.hidden_dropout(self.activation(self.linear1(x)))
        x = self.dropout(self.linear2(x))

        x = x + residual

        x = self.normalization(x)

        return x
