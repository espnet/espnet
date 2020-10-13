"""Set of methods to build customizable encoder."""

from typing import Any
from typing import Dict
from typing import List
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import (
    EncoderLayer as ConformerEncoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transducer.tdnn import TDNN
from espnet.nets.pytorch_backend.transformer.encoder_layer import (
    EncoderLayer as TransformerEncoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet2.asr.custom.utils import config_verification
from espnet2.asr.custom.utils import get_positional_encoding_class
from espnet2.asr.custom.utils import get_positionwise_class
from espnet2.asr.custom.utils import get_self_attention_class
from espnet2.asr.custom.vgg2l import VGG2L


# layer type: mandatory parameters
supported_layers = {
    "input": {
        "conv2d": "hidden_size",
        "conv2d6": "hidden_size",
        "conv2d8": "hidden_size",
        "embed": "hidden_size",
        "linear": "hidden_size",
        "vgg2l": "hidden_size",
    },
    "body": {
        "conformer": "hidden_size",
        "tdnn": ("input_size", "output_size"),
        "transformer": "hidden_size",
    },
}


def build_transformer_layer(
    positional_encoding,
    positionwise,
    self_attention,
    hidden_size: int = 320,
    linear_units: int = 320,
    attention_heads: int = 4,
    positionwise_activation_type: str = "relu",
    positionwise_convolution_kernel_size: int = 3,
    dropout_rate: float = 0.0,
    positionwise_dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    normalize_before: bool = True,
    concat_after: bool = False,
) -> torch.nn.Module:
    """Build transformer layer."""
    assert check_argument_types()

    positionwise_activation = get_activation(positionwise_activation_type)

    if getattr(positionwise, "__name__", None) == PositionwiseFeedForward.__name__:
        positionwise_args = (
            hidden_size,
            linear_units,
            positionwise_dropout_rate,
            positionwise_activation,
        )
    else:
        positionwise_args = (
            hidden_size,
            linear_units,
            positionwise_convolution_kernel_size,
            positionwise_dropout_rate,
        )

    self_attention_args = (
        attention_heads,
        hidden_size,
        attention_dropout_rate,
    )

    return lambda: TransformerEncoderLayer(
        hidden_size,
        self_attention(*self_attention_args),
        positionwise(*positionwise_args),
        dropout_rate,
        normalize_before,
        concat_after,
    )


def build_conformer_layer(
    positional_encoding,
    positionwise,
    self_attention,
    hidden_size: int = 320,
    linear_units: int = 320,
    attention_heads: int = 4,
    positionwise_activation_type: str = "swish",
    positionwise_convolution_kernel_size: int = 3,
    macaron_style: bool = False,
    use_cnn_module: bool = False,
    cnn_module_kernel: int = 31,
    dropout_rate: float = 0.0,
    positionwise_dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    normalize_before: bool = True,
    concat_after: bool = False,
) -> torch.nn.Module:
    """Build conformer layer."""
    assert check_argument_types()

    positionwise_activation = get_activation(positionwise_activation_type)

    if getattr(positionwise, "__name__", None) == PositionwiseFeedForward.__name__:
        positionwise_args = (
            hidden_size,
            linear_units,
            positionwise_dropout_rate,
            positionwise_activation,
        )
    else:
        positionwise_args = (
            hidden_size,
            linear_units,
            positionwise_convolution_kernel_size,
            positionwise_dropout_rate,
        )

    self_attention_args = (
        attention_heads,
        hidden_size,
        attention_dropout_rate,
    )

    convolution = ConvolutionModule
    convolution_args = (hidden_size, cnn_module_kernel, positionwise_activation)

    return lambda: ConformerEncoderLayer(
        hidden_size,
        self_attention(*self_attention_args),
        positionwise(*positionwise_args),
        positionwise(*positionwise_args) if macaron_style else None,
        convolution(*convolution_args) if use_cnn_module else None,
        dropout_rate,
        normalize_before,
        concat_after,
    )


def build_tdnn_layer(
    input_size: int = 320,
    output_size: int = 320,
    context_size: int = 5,
    dilation: int = 1,
    stride: int = 1,
    use_batch_norm: bool = False,
    use_relu: bool = True,
    dropout_rate: float = 0.0,
):
    """Build tdnn layer."""
    assert check_argument_types()

    return lambda: TDNN(
        input_size,
        output_size,
        ctx_size=context_size,
        dilation=dilation,
        stride=stride,
        batch_norm=use_batch_norm,
        relu=use_relu,
        dropout_rate=dropout_rate,
    )


def build_input_layer(
    input_size: int,
    positional_encoding,
    hidden_size: int = 320,
    layer_type: str = "embed",
    use_positional_encoding: bool = True,
    dropout_rate: float = 0.0,
    positional_encoding_dropout_rate: float = 0.0,
    padding_idx: int = -1,
) -> torch.nn.Module:
    """Build input layer."""
    assert check_argument_types()

    if use_positional_encoding:
        pos_enc_class = positional_encoding(
            hidden_size, positional_encoding_dropout_rate
        )
    else:
        pos_enc_class = None

    if layer_type == "linear":
        embed = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            pos_enc_class,
        )
    elif layer_type == "conv2d":
        embed = Conv2dSubsampling(input_size, hidden_size, dropout_rate, pos_enc_class)
    elif layer_type == "conv2d6":
        embed = Conv2dSubsampling6(input_size, hidden_size, dropout_rate, pos_enc_class)
    elif layer_type == "conv2d8":
        embed = Conv2dSubsampling8(input_size, hidden_size, dropout_rate, pos_enc_class)
    elif layer_type == "vgg2l":
        embed = VGG2L(input_size, hidden_size, dropout_rate, pos_enc_class)
    elif layer_type == "embed":
        embed = torch.nn.Sequential(
            torch.nn.Embedding(input_size, hidden_size, padding_idx=padding_idx),
            pos_enc_class,
        )
    elif layer_type is None:
        embed = torch.nn.Sequential(pos_enc_class)

    return embed


def build_encoder(
    input_size: int,
    architecture: List[Dict[str, Any]],
    positional_encoding_type: str = "abs_pos",
    positionwise_type: str = "linear",
    self_attention_type: str = "self_attn",
    repeat: int = 0,
    padding_idx: int = -1,
) -> Union[torch.nn.Module, MultiSequential, int]:
    """Build encoder based on architecture specification.

    Args:
        input_size: Input dimension
        architecture: Encoder architecture configuration
        positional_encoding_type: 'abs_pos', 'scaled_abs_pos', 'rel_pos'
        positionwise_type: 'linear', 'conv1d', 'conv1d-linear'
        self_attention_type: 'self_attn', 'rel_self_attn'
        repeat: Number of times specified architecture should be repeated
        padding_idx: Index for embedding padding

    """
    output_size = config_verification(
        input_size, supported_layers, architecture, repeat
    )

    pos_enc_class = get_positional_encoding_class(
        positional_encoding_type, self_attention_type
    )
    pw_class = get_positionwise_class(positionwise_type)
    self_att_class = get_self_attention_class(self_attention_type)
    arch_classes = (pos_enc_class, pw_class, self_att_class)

    input_layer = build_input_layer(
        input_size, pos_enc_class, padding_idx=padding_idx, **architecture[0]
    )

    architecture_modules = []
    for layer_conf in architecture[1:]:
        layer_type = layer_conf["layer_type"]
        layer_args = {x: layer_conf[x] for x in layer_conf if x != "layer_type"}

        if layer_type == "conformer":
            module = build_conformer_layer(*arch_classes, **layer_args)
        elif layer_type == "tdnn":
            module = build_tdnn_layer(**layer_args)
        elif layer_type == "transformer":
            module = build_transformer_layer(*arch_classes, **layer_args)

        architecture_modules.append(module)

    if repeat > 1:
        architecture_modules = architecture_modules * repeat

    return (
        input_layer,
        MultiSequential(*[mod() for mod in architecture_modules]),
        output_size,
    )
