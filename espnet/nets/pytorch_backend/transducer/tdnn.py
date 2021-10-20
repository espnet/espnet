"""TDNN modules definition for custom encoder."""

from typing import Tuple
from typing import Union

import torch


class TDNN(torch.nn.Module):
    """TDNN module with symmetric context.

    Args:
        idim: Input dimension.
        odim: Output dimension.
        ctx_size: Size of the context window.
        stride: Stride of the sliding blocks.
        dilation: Control the stride of elements within the neighborhood.
        batch_norm: Whether to apply batch normalization.
        relu: Whether to use a final non-linearity layer (ReLU).

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
        sequence: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Forward TDNN for custom encoder.

        Args:
            sequence: TDNN input sequences.
                        (B, T, D_in) or ((B, T, D_in), (B, T, D_att))
                          or ((B, T, D_in), (B, 2 * (T - 1), D_att))
            mask: Mask of TDNN input sequences. (B, 1, T)

        Returns:
            sequence: TDNN output sequences.
                        (B, sub(T), D_out)
                          or ((B, sub(T), D_out), (B, sub(T), att_dim))
                          or ((B, T, D_out), (B, 2 * (sub(T) - 1), D_att)
            mask: Mask of TDNN output sequences. (B, 1, sub(T))

        """
        if isinstance(sequence, tuple):
            sequence, pos_emb = sequence[0], sequence[1]
        else:
            sequence, pos_emb = sequence, None

        sequence = sequence.transpose(1, 2)

        # The bidirect_pos is used to distinguish legacy_rel_pos and rel_pos in
        # Conformer model. Note the `legacy_rel_pos` will be deprecated in the future.
        # Details can be found in https://github.com/espnet/espnet/pull/2816.
        if pos_emb is not None and pos_emb.size(1) == (2 * sequence.size(2)) - 1:
            bidir_pos_emb = True
        else:
            bidir_pos_emb = False

        sequence = self.tdnn(sequence)

        if self.relu:
            sequence = self.relu_func(sequence)

        sequence = self.dropout(sequence)

        if self.batch_norm:
            sequence = self.bn(sequence)

        sequence = sequence.transpose(1, 2)

        return self.create_outputs(sequence, pos_emb, mask, bidir_pos_emb)

    def create_outputs(
        self,
        sequence: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor,
        bidir_pos_emb: bool,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Create outputs with subsampled version of pos_emb and masks.

        Args:
            sequence: TDNN output sequences. (B, sub(T), D_out)
            pos_emb: Input positional embedding.
                     (B, T, att_dim) or (B, 2 * (T - 1), D_att)
            mask: Mask of TDNN input sequences. (B, 1, T)
            bidir_pos_emb: Whether to use bidirectional positional embedding.

        Returns:
            sequence: TDNN output sequences. (B, sub(T), D_out)
            pos_emb: Output positional embedding.
                     (B, sub(T), D_att) or (B, 2 * (sub(T) - 1), D_att)
            mask: Mask of TDNN output sequences. (B, 1, sub(T))

        """
        sub = (self.ctx_size - 1) * self.dilation

        if mask is not None:
            if sub != 0:
                mask = mask[:, :, :-sub]

            mask = mask[:, :, :: self.stride]

        if pos_emb is not None:
            # If the bidirect_pos is true, the pos_emb will include both positive and
            # negative embeddings. Refer to https://github.com/espnet/espnet/pull/2816.
            if bidir_pos_emb:
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

            return (sequence, pos_emb), mask

        return sequence, mask
