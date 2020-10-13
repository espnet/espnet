"""Set of methods to build customizable decoder."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from espnet2.asr.custom.causal_conv1d import CausalConv1d
from espnet2.asr.custom.utils import config_verification
from espnet2.asr.custom.utils import get_lightweight_dynamic_convolution_class
from espnet2.asr.custom.utils import get_positional_encoding_class
from espnet2.asr.custom.utils import get_positionwise_class
from espnet2.asr.custom.utils import get_self_attention_class


# layer type: mandatory parameters
supported_layers = {
    "input": {
        "embed": "hidden_size",
        "linear": "hidden_size",
    },
    "body": {
        "causal_conv1d": ("input_size", "output_size"),
        "dynamic_conv": "hidden_size",
        "dynamic_conv2d": "hidden_size",
        "lightweight_conv": "hidden_size",
        "lightweight_conv2d": "hidden_size",
        "transformer": "hidden_size",
    },
}


def build_lightweight_dynamic_convolution_layer(
    lightweight_dynamic_convolution,
    positional_encoding,
    positionwise,
    self_attention,
    hidden_size: int = 320,
    linear_units: int = 320,
    attention_heads: int = 4,
    conv_wshare: int = 4,
    conv_kernel_length: int = 11,
    conv_usebias: int = False,
    positionwise_activation_type: str = "relu",
    dropout_rate: float = 0.0,
    self_attention_dropout_rate: float = 0.0,
    src_attention_dropout_rate: float = 0.0,
    positionwise_dropout_rate: float = 0.0,
    normalize_before: bool = True,
    concat_after: bool = False,
) -> torch.nn.Module:
    """Build lightweight/dynamic convolution layer."""
    assert check_argument_types()

    positionwise_activation = get_activation(positionwise_activation_type)
    positionwise_args = (
        hidden_size,
        linear_units,
        positionwise_dropout_rate,
        positionwise_activation,
    )

    src_attention_args = (
        attention_heads,
        hidden_size,
        src_attention_dropout_rate,
    )

    return lambda: DecoderLayer(
        hidden_size,
        lightweight_dynamic_convolution(
            wshare=conv_wshare,
            n_feat=hidden_size,
            dropout_rate=self_attention_dropout_rate,
            kernel_size_str=str(conv_kernel_length),
            lnum=0,
            use_kernel_mask=True,
            use_bias=conv_usebias,
        ),
        self_attention(*src_attention_args),
        positionwise(*positionwise_args),
        dropout_rate,
        normalize_before,
        concat_after,
    )


def build_transformer_layer(
    positional_encoding,
    positionwise,
    self_attention,
    hidden_size: int = 320,
    linear_units: int = 320,
    attention_heads: int = 4,
    positionwise_activation_type: str = "relu",
    dropout_rate: float = 0.0,
    self_attention_dropout_rate: float = 0.0,
    src_attention_dropout_rate: float = 0.0,
    positionwise_dropout_rate: float = 0.0,
    normalize_before: bool = True,
    concat_after: bool = False,
) -> torch.nn.Module:
    """Build transformer layer."""
    assert check_argument_types()

    positionwise_activation = get_activation(positionwise_activation_type)
    positionwise_args = (
        hidden_size,
        linear_units,
        positionwise_dropout_rate,
        positionwise_activation,
    )

    self_attention_args = (
        attention_heads,
        hidden_size,
        self_attention_dropout_rate,
    )
    src_attention_args = (
        attention_heads,
        hidden_size,
        src_attention_dropout_rate,
    )

    return lambda: DecoderLayer(
        hidden_size,
        self_attention(*self_attention_args),
        self_attention(*src_attention_args),
        positionwise(*positionwise_args),
        dropout_rate,
        normalize_before,
        concat_after,
    )


def build_causal_convolution(
    input_size: int = 320,
    output_size: int = 320,
    kernel_size: int = 1,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
):
    """Build causal conv1d layer."""
    assert check_argument_types()

    return lambda: CausalConv1d(
        input_size,
        output_size,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def build_input_layer(
    input_size: int,
    positional_encoding,
    hidden_size: int = 320,
    layer_type: str = "embed",
    positional_encoding_type: str = "abs_pos",
    dropout_rate: float = 0.0,
    positional_encoding_dropout_rate: float = 0.0,
    padding_idx: Optional[int] = None,
) -> torch.nn.Module:
    """Build input layer."""
    assert check_argument_types()

    if layer_type == "linear":
        embed = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            positional_encoding(hidden_size, positional_encoding_dropout_rate),
        )
    elif layer_type == "embed":
        embed = torch.nn.Sequential(
            torch.nn.Embedding(input_size, hidden_size, padding_idx=padding_idx),
            positional_encoding(hidden_size, positional_encoding_dropout_rate),
        )
    elif layer_type is None:
        embed = torch.nn.Sequential(
            positional_encoding(hidden_size, positional_encoding_dropout_rate)
        )

    return embed


def build_decoder(
    input_size: int,
    architecture: List[Dict[str, Any]],
    positional_encoding_type: str = "abs_pos",
    positionwise_type: str = "linear",
    self_attention_type: str = "self_attn",
    lightweight_dynamic_convolution_type: str = "lightweight_conv",
    repeat: int = 0,
    padding_idx: Optional[int] = None,
) -> Union[torch.nn.Module, MultiSequential, int]:
    """Build decoder based on architecture specification.

    Args:
        input_size: Input dimension
        architecture: Decoder architecture configuration
        positional_encoding_type: 'abs_pos', 'scaled_abs_pos', 'rel_pos'
        positionwise_type: 'linear', 'conv1d', 'conv1d-linear'
        self_attention_type: 'self_attn', 'rel_self_attn'
        lightweight_dynamic_convolution_type:
            'lightweight_conv', 'dynamic_conv',
            'lightweight_conv2d', 'dynamic_conv2d'
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
    ld_conv_class = get_lightweight_dynamic_convolution_class(
        lightweight_dynamic_convolution_type
    )
    arch_classes = (pos_enc_class, pw_class, self_att_class)

    input_layer = build_input_layer(
        input_size, pos_enc_class, padding_idx=padding_idx, **architecture[0]
    )

    architecture_modules = []
    for layer_conf in architecture[1:]:
        layer_type = layer_conf["layer_type"]
        layer_args = {x: layer_conf[x] for x in layer_conf if x != "layer_type"}

        if layer_type == "causal_conv1d":
            module = build_causal_convolution(**layer_args)
        if layer_type == "transformer":
            module = build_transformer_layer(*arch_classes, **layer_args)
        elif layer_type.startswith(("lightweight", "dynamic")):
            module = build_lightweight_dynamic_convolution_layer(
                ld_conv_class, *arch_classes, **layer_args
            )

        architecture_modules.append(module)

    if repeat > 1:
        architecture_modules = architecture_modules * repeat

    return (
        input_layer,
        MultiSequential(*[mod() for mod in architecture_modules]),
        output_size,
    )
