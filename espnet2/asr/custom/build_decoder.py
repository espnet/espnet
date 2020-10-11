"""Set of methods to build customizable decoder."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transducer.causal_conv1d import CausalConv1d
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from espnet2.asr.custom.utils import get_lightweight_dynamic_convolution_class
from espnet2.asr.custom.utils import get_positional_encoding_class
from espnet2.asr.custom.utils import get_positionwise_class
from espnet2.asr.custom.utils import get_self_attention_class


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


def verify_layers_io(input_size: int, architecture: List[Dict[str, Any]]):
    """Verify defined layers input-output are valid before creation.

    Args:
        input_size: Input layer size
        architecture: Decoder architecture configuration

    Returns:
        (): if architecture is valid, return output size
    """
    check_io = []

    for i, layer_conf in enumerate(architecture):
        if "layer_type" in layer_conf:
            layer_type = layer_conf["layer_type"]
        else:
            raise ValueError(
                "layer_type is not defined in the " + str(i + 1) + "th layer."
            )

        if layer_type in ["embed", "linear"]:
            if "hidden_size" not in layer_conf:
                raise ValueError(
                    "Layer " + str(i + 1) + ": Format is: "
                    "{'layer_type: embed|linear', 'hidden_size': int, [...]}"
                )
            check_io.append((input_size, layer_conf["hidden_size"]))
        elif layer_type in [
            "lightweight",
            "dynamic",
            "lightweight2d",
            "dynamic2d",
            "transformer",
        ]:
            if not {"hidden_size", "linear_units", "attention_heads"}.issubset(
                layer_conf
            ):
                raise ValueError(
                    "Layer " + str(i + 1) + ": Format is: "
                    "{'layer_type': transformer|lightweight*|dynamic*, "
                    "'hidden_size': int, 'linear_units': int, "
                    "'attention_heads': int, [...]}"
                )
            check_io.append((layer_conf["hidden_size"], layer_conf["hidden_size"]))
        elif layer_type == "causal_conv1d":
            if not {"input_size", "output_size"}.issubset(layer_conf):
                raise ValueError(
                    "Layer " + str(i + 1) + ": Format is: "
                    "{'layer_type': causal_conv1d, 'input_size': int, "
                    "'output_size': int, [...]}"
                )
            check_io.append((layer_conf["input_size"], layer_conf["output_size"]))
        else:
            raise NotImplementedError(
                "Layer "
                + str(i + 1)
                + ": layer_type "
                + layer_type
                + " is not supported."
            )

    for i in range(1, len(check_io)):
        if check_io[(i - 1)][1] != check_io[i][0]:
            raise ValueError(
                "Output-Input mismatch between layers " + str(i) + " and " + str(i + 1)
            )

    return check_io[-1][1]


def build_decoder(
    input_size: int,
    architecture: List[Dict[str, Any]],
    positional_encoding_type: str = "abs_pos",
    positionwise_type: str = "linear",
    self_attention_type: str = "self_attn",
    lightweight_dynamic_convolution_type: str = "lightweight",
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
            'lightweight', 'dynamic', 'lightweight2d', 'dynamic2d'
        repeat: Number of times specified architecture should be repeated
        padding_idx: Index for embedding padding

    """
    output_size = verify_layers_io(input_size, architecture)

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
