"""Normalization modules for X-former blocks."""

import torch


class BasicNorm(torch.nn.Module):
    """BasicNorm module definition.

    Args:
        number_channels: The number of channels.
        channel_dim: The channel dimension to normalize, expressed as an offset.
                     A negative value denotes a left offset, from the last dimension.
                     A positive value denotes a right offset, from the first dimension.
        eps: Initial epsilon value.
        learn_eps: Whether to learn epsilon value or keep initial value.

    Reference: https://github.com/k2-fsa/icefall/pull/288

    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,
        eps: float = 0.25,
    ) -> None:
        """Construct a BasicNorm object."""
        super().__init__()

        self.eps = torch.nn.Parameter(torch.tensor(eps).log().detach())

        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute basic normalization.

        Args:
            x: Input sequences. (B, T, D_hidden)

        Returns:
            : Output sequences. (B, T, D_hidden)

        """
        scales = (
            torch.mean(x.pow(2), dim=self.channel_dim, keepdim=True) + self.eps.exp()
        ) ** -0.5

        return x * scales
