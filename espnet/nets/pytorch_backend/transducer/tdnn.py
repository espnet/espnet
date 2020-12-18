"""TDNN modules definition for transformer encoder."""

from typing import Tuple
from typing import Union

import torch


class TDNN(torch.nn.Module):
    """TDNN implementation with symmetric context.

    Args:
        idim: Dimension of inputs
        odim: Dimension of outputs
        ctx_size: Size of context window
        stride: Stride of the sliding blocks
        dilation: Parameter to control the stride of
                  elements within the neighborhood
        batch_norm: Whether to use batch normalization
        relu: Whether to use non-linearity layer (ReLU)

    """

    def __init__(
        self,
        idim: int,
        odim: int,
        ctx_size: int = 5,
        dilation: int = 1,
        stride: int = 1,
        batch_norm: bool = False,
        relu: bool = True,
        dropout_rate: float = 0.0,
    ):
        """Construct a TDNN object."""
        super().__init__()

        self.idim = idim
        self.odim = odim

        self.ctx_size = ctx_size
        self.stride = stride
        self.dilation = dilation

        self.batch_norm = batch_norm
        self.relu = relu

        self.tdnn = torch.nn.Conv1d(
            idim, odim, ctx_size, stride=stride, dilation=dilation
        )

        if self.relu:
            self.relu_func = torch.nn.ReLU()

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(odim)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        masks: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Forward TDNN.

        Args:
            x_input: Input tensor (B, T, idim) or ((B, T, idim), (B, T, att_dim))
            masks: Input mask (B, 1, T)

        Returns:
            x_output: Output tensor (B, sub(T), odim)
                          or ((B, sub(T), odim), (B, sub(T), att_dim))
            mask: Output mask (B, 1, sub(T))

        """
        if isinstance(x_input, tuple):
            xs, pos_emb = x_input[0], x_input[1]
        else:
            xs, pos_emb = x_input, None

        xs = xs.transpose(1, 2)
        xs = self.tdnn(xs)

        if self.relu:
            xs = self.relu_func(xs)

        xs = self.dropout(xs)

        if self.batch_norm:
            xs = self.bn(xs)

        xs = xs.transpose(1, 2)

        return self.create_outputs(xs, pos_emb, masks)

    def create_outputs(
        self, xs: torch.Tensor, pos_emb: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Create outputs with subsampled version of pos_emb and masks.

        Args:
            xs: Output tensor (B, sub(T), odim)
            pos_emb: Input positional embedding tensor (B, T, att_dim)
            masks: Input mask (B, 1, T)

        Returns:
            xs: Output tensor (B, sub(T), odim)
            pos_emb: Output positional embedding tensor (B, sub(T), att_dim)
            masks: Output mask (B, 1, sub(T))

        """
        sub = (self.ctx_size - 1) * self.dilation

        if masks is not None:
            if sub != 0:
                masks = masks[:, :, :-sub]

            masks = masks[:, :, :: self.stride]

        if pos_emb is not None:
            if sub != 0:
                pos_emb = pos_emb[:, :-sub, :]

            pos_emb = pos_emb[:, :: self.stride, :]

            return (xs, pos_emb), masks

        return xs, masks
