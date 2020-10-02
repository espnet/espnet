"""Encoder definition for transformer-transducer models."""

import torch

from espnet.nets.pytorch_backend.transducer.blocks import build_blocks
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        idim (int): input dim
        enc_arch (list): list of encoder blocks (type and parameters)
        input_layer (str): input layer type
        repeat_block (int): repeat provided block N times if N > 1
        self_attn_type (str): type of self-attention
        positional_encoding_type (str): positional encoding type
        positionwise_layer_type (str): linear
        positionwise_activation_type (str): positionwise activation type
        conv_mod_activation_type (str): convolutional module activation type
        normalize_before (bool): whether to use layer_norm before the first block
        padding_idx (int): padding_idx for embedding input layer (if specified)

    """

    def __init__(
        self,
        idim,
        enc_arch,
        input_layer="linear",
        repeat_block=0,
        self_attn_type="selfattn",
        positional_encoding_type="abs_pos",
        positionwise_layer_type="linear",
        positionwise_activation_type="relu",
        conv_mod_activation_type="relu",
        normalize_before=True,
        padding_idx=-1,
    ):
        """Construct an Transformer encoder object."""
        super().__init__()

        self.embed, self.encoders, self.enc_out = build_blocks(
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
            padding_idx=padding_idx,
        )

        self.normalize_before = normalize_before

        if self.normalize_before:
            self.after_norm = LayerNorm(self.enc_out)

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): input tensor
            masks (torch.Tensor): input mask

        Returns:
            xs (torch.Tensor): position embedded input
            mask (torch.Tensor): position embedded mask

        """
        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        xs, masks = self.encoders(xs, masks)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks
