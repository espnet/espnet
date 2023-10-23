from typing import List, Optional, Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import TransformerEncoder
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

class EncoderLayerPipe(EncoderLayer):
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__(
            size,
            self_attn,
            feed_forward,
            dropout_rate,
            normalize_before,
            concat_after,
            stochastic_depth_rate,
        )
    
    def forward(self, inputs):
        x, tup = inputs
        cache=None

        if isinstance(tup, tuple):
            mask, cache = tup
        else:
            mask = tup

        x, mask = super().forward(x, mask, cache)
        return (x, mask, cache)

class TransformerEncoderPipe(TransformerEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        layer_drop_rate: float = 0.0,
    ): 

        assert check_argument_types()

        # wont support interctc
        super().__init__(
            input_size,
            output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            attention_dropout_rate,
            input_layer,
            pos_enc_class,
            normalize_before,
            concat_after,
            positionwise_layer_type,
            positionwise_conv_kernel_size,
            padding_idx,
            interctc_layer_idx: List[int] = [],
            interctc_use_conditioning: bool = False,
            layer_drop_rate,
        )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayerPipe(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )
    
    def to_layers(self):
        return [*self.encoders]


        
