"""Encoder definition for transformer-transducer models."""

import torch

from espnet.nets.pytorch_backend.transducer.blocks import build_blocks
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    Args:
        idim (int): input dim
        enc_arch (List[dict]): list of layer definitions
        input_layer (str): input layer type
        repeat_block (int): if N > 1, repeat block N times
        pos_enc_class (class): PositionalEncoding or ScaledPositionalEncoding
        positionwise_layer_type (str): linear of conv1d
        positionwise_conv_kernel_size (int) : kernel size of positionwise conv1d layer
        dropout_rate (float): dropout rate
        attention_dropout_rate (float): dropout rate in attention
        positional_dropout_rate (float): dropout rate after adding positional encoding
        normalize_before (bool): whether to use layer_norm before the first block
        concat_after (bool): whether to concat attention layer's input and output
        padding_idx (int): padding_idx for input_layer=embed

    """

    def __init__(
        self,
        idim,
        enc_arch,
        input_layer="linear",
        repeat_block=0,
        pos_enc_class=PositionalEncoding,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        dropout_rate=0.1,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        normalize_before=True,
        concat_after=False,
        padding_idx=-1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        self.embed, self.encoders, self.enc_out = build_blocks(
            idim,
            input_layer,
            enc_arch,
            EncoderLayer,
            repeat_block=repeat_block,
            pos_enc_class=pos_enc_class,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            att_dropout_rate=attention_dropout_rate,
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

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        Args:
            xs (torch.Tensor): input tensor
            masks (torch.Tensor): input mask
            cache (List[torch.Tensor]): cache tensors

        Returns:
            xs (torch.Tensor): position embedded input tensor
            masks (torch.Tensor): position embedded input mask
            new_cache (List[torch.Tensor]): position embedded cache

        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
