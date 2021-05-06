"""TDNN modules definition for transformer encoder."""

import logging
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
            or ((B, T, idim), (B, 2*T-1, att_dim))
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

        # The bidirect_pos is used to distinguish legacy_rel_pos and rel_pos in
        # Conformer model. Note the `legacy_rel_pos` will be deprecated in the future.
        # Details can be found in https://github.com/espnet/espnet/pull/2816.
        if pos_emb is not None and pos_emb.size(1) == 2 * xs.size(1) - 1:
            logging.warning("Using bidirectional relative postitional encoding.")
            bidirect_pos = True
        else:
            bidirect_pos = False

        xs = xs.transpose(1, 2)
        xs = self.tdnn(xs)

        if self.relu:
            xs = self.relu_func(xs)

        xs = self.dropout(xs)

        if self.batch_norm:
            xs = self.bn(xs)

        xs = xs.transpose(1, 2)

        return self.create_outputs(xs, pos_emb, masks, bidirect_pos=bidirect_pos)

    def create_outputs(
        self,
        xs: torch.Tensor,
        pos_emb: torch.Tensor,
        masks: torch.Tensor,
        bidirect_pos: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Create outputs with subsampled version of pos_emb and masks.

        Args:
            xs: Output tensor (B, sub(T), odim)
            pos_emb: Input positional embedding tensor (B, T, att_dim)
            or (B, 2*T-1, att_dim)
            masks: Input mask (B, 1, T)
            bidirect_pos: whether to use bidirectional positional embedding

        Returns:
            xs: Output tensor (B, sub(T), odim)
            pos_emb: Output positional embedding tensor (B, sub(T), att_dim)
            or (B, 2*sub(T)-1, att_dim)
            masks: Output mask (B, 1, sub(T))

        """
        sub = (self.ctx_size - 1) * self.dilation

        if masks is not None:
            if sub != 0:
                masks = masks[:, :, :-sub]

            masks = masks[:, :, :: self.stride]

        if pos_emb is not None:
            # If the bidirect_pos is true, the pos_emb will include both positive and
            # negative embeddings. Refer to https://github.com/espnet/espnet/pull/2816.
            if bidirect_pos:
                pos_emb_positive = pos_emb[:, : pos_emb.size(1) // 2 + 1, :]
                pos_emb_negative = pos_emb[:, pos_emb.size(1) // 2 :, :]

                if sub != 0:
                    pos_emb_positive = pos_emb_positive[:, :-sub, :]
                    pos_emb_negative = pos_emb_negative[:, :-sub, :]

                pos_emb_positive = pos_emb_positive[:, :: self.stride, :]
                pos_emb_negative = pos_emb_negative[:, :: self.stride, :]
                pos_emb = torch.cat(
                    [pos_emb_positive, pos_emb_negative[:, 1:, :]], dim=1
                )
            else:
                if sub != 0:
                    pos_emb = pos_emb[:, :-sub, :]

                pos_emb = pos_emb[:, :: self.stride, :]

            return (xs, pos_emb), masks

        return xs, masks
