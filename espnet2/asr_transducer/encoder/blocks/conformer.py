"""Conformer block for Transducer encoder."""

from typing import Tuple
from typing import Union

import torch


class Conformer(torch.nn.Module):
    """Conformer module definition.

    Args:
        size: Input/output dimension.
        self_att: Self-attention module instance.
        feed_forward: Feed-forward module instance.
        feed_forward_macaron: Feed-forward module instance for macaron network.
        conv_mod: Convolution module instance.
        dropout_rate: Dropout rate.
        eps_layer_norm: Epsilon value for LayerNorm.

    """

    def __init__(
        self,
        size: int,
        self_att: torch.nn.Module,
        feed_forward: torch.nn.Module,
        feed_forward_macaron: torch.nn.Module,
        conv_mod: torch.nn.Module,
        dropout_rate: float = 0.0,
        eps_layer_norm: float = 1e-12,
    ):
        """Construct a Conformer object."""

        super().__init__()

        self.self_att = self_att

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron

        self.conv_mod = conv_mod

        self.norm_feed_forward = torch.nn.LayerNorm(size, eps_layer_norm)
        self.norm_multihead_att = torch.nn.LayerNorm(size, eps_layer_norm)

        if feed_forward_macaron is not None:
            self.norm_macaron = torch.nn.LayerNorm(size, eps_layer_norm)
            self.feed_forward_scale = 0.5
        else:
            self.feed_forward_scale = 1.0

        if self.conv_mod is not None:
            self.norm_conv = torch.nn.LayerNorm(size, eps_layer_norm)
            self.norm_final = torch.nn.LayerNorm(size, eps_layer_norm)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.size = size

    def forward(
        self,
        sequence: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        cache: torch.Tensor = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Encode input sequences.

        Args:
            sequence: Conformer input sequences.
                     (B, T, D_emb) or ((B, T, D_emb), (1, T, D_emb))
            mask: Mask of input sequences. (B, T)
            cache: Conformer cache. (B, T-1, D_hidden)

        Returns:
            sequence: Conformer output sequences.
               (B, T, D_enc) or ((B, T, D_enc), (1, T, D_enc))
            mask: Mask of output sequences. (B, T)

        """
        if isinstance(sequence, tuple):
            sequence, pos_emb = sequence[0], sequence[1]
        else:
            sequence, pos_emb = sequence, None

        if self.feed_forward_macaron is not None:
            residual = sequence

            sequence = self.norm_macaron(sequence)
            sequence = residual + self.feed_forward_scale * self.dropout(
                self.feed_forward_macaron(sequence)
            )

        residual = sequence
        sequence = self.norm_multihead_att(sequence)
        x_q = sequence

        if pos_emb is not None:
            sequence_att = self.self_att(x_q, sequence, sequence, pos_emb, mask)
        else:
            sequence_att = self.self_att(x_q, sequence, sequence, mask)

        sequence = residual + self.dropout(sequence_att)

        if self.conv_mod is not None:
            residual = sequence

            sequence = self.norm_conv(sequence)
            sequence = residual + self.dropout(self.conv_mod(sequence))

        residual = sequence

        sequence = self.norm_feed_forward(sequence)
        sequence = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward(sequence)
        )

        if self.conv_mod is not None:
            sequence = self.norm_final(sequence)

        if pos_emb is not None:
            return (sequence, pos_emb), mask

        return sequence, mask
