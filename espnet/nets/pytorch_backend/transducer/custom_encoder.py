"""Cutom encoder definition for transducer models."""

from typing import List, Tuple, Union

import torch

from espnet.nets.pytorch_backend.transducer.blocks import build_blocks
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


class CustomEncoder(torch.nn.Module):
    """Custom encoder module for transducer models.

    Args:
        idim: Input dimension.
        enc_arch: Encoder block architecture (type and parameters).
        input_layer: Input layer type.
        repeat_block: Number of times blocks_arch is repeated.
        self_attn_type: Self-attention type.
        positional_encoding_type: Positional encoding type.
        positionwise_layer_type: Positionwise layer type.
        positionwise_activation_type: Positionwise activation type.
        conv_mod_activation_type: Convolutional module activation type.
        aux_enc_output_layers: Layer IDs for auxiliary encoder output sequences.
        input_layer_dropout_rate: Dropout rate for input layer.
        input_layer_pos_enc_dropout_rate: Dropout rate for input layer pos. enc.
        padding_idx: Padding symbol ID for embedding layer.

    """

    def __init__(
        self,
        idim: int,
        enc_arch: List,
        input_layer: str = "linear",
        repeat_block: int = 1,
        self_attn_type: str = "selfattn",
        positional_encoding_type: str = "abs_pos",
        positionwise_layer_type: str = "linear",
        positionwise_activation_type: str = "relu",
        conv_mod_activation_type: str = "relu",
        aux_enc_output_layers: List = [],
        input_layer_dropout_rate: float = 0.0,
        input_layer_pos_enc_dropout_rate: float = 0.0,
        padding_idx: int = -1,
    ):
        """Construct an CustomEncoder object."""
        super().__init__()

        (
            self.embed,
            self.encoders,
            self.enc_out,
            self.conv_subsampling_factor,
        ) = build_blocks(
            "encoder",
            idim,
            input_layer,
            enc_arch,
            repeat_block=repeat_block,
            self_attn_type=self_attn_type,
            positional_encoding_type=positional_encoding_type,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_activation_type=positionwise_activation_type,
            conv_mod_activation_type=conv_mod_activation_type,
            input_layer_dropout_rate=input_layer_dropout_rate,
            input_layer_pos_enc_dropout_rate=input_layer_pos_enc_dropout_rate,
            padding_idx=padding_idx,
        )

        self.after_norm = LayerNorm(self.enc_out)

        self.n_blocks = len(enc_arch) * repeat_block

        self.aux_enc_output_layers = aux_enc_output_layers

    def forward(
        self, feats: torch.Tensor, mask: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """Encode feature sequences.

        Args:
            feats: Feature sequences. (B, F, D_feats)
            feats_mask: Feature mask sequences. (B, 1, F)

        Returns:
            enc_out: Encoder output sequences. (B, T, D_enc) with/without
                     Auxiliary encoder output sequences. (B, T, D_enc_aux)
            enc_out_mask: Mask for encoder output sequences. (B, 1, T) with/without
                          Mask for auxiliary encoder output sequences. (B, T, D_enc_aux)

        """
        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            enc_out, mask = self.embed(feats, mask)
        else:
            enc_out = self.embed(feats)

        if self.aux_enc_output_layers:
            aux_custom_outputs = []
            aux_custom_lens = []

            for b in range(self.n_blocks):
                enc_out, mask = self.encoders[b](enc_out, mask)

                if b in self.aux_enc_output_layers:
                    if isinstance(enc_out, tuple):
                        aux_custom_output = enc_out[0]
                    else:
                        aux_custom_output = enc_out

                    aux_custom_outputs.append(self.after_norm(aux_custom_output))
                    aux_custom_lens.append(mask)

        else:
            enc_out, mask = self.encoders(enc_out, mask)

        if isinstance(enc_out, tuple):
            enc_out = enc_out[0]

        enc_out = self.after_norm(enc_out)

        if self.aux_enc_output_layers:
            return (enc_out, aux_custom_outputs), (mask, aux_custom_lens)

        return enc_out, mask
