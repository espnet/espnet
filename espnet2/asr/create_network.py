"""Set of methods to create customizable network."""

from typing import Any
from typing import Dict
from typing import List
from typing import Union

import torch

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import (
    EncoderLayer as ConformerEncoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import (
    EncoderLayer as TransformerEncoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


def verify_layers_io(input_size: int, architecture: List[Dict[str, Any]]):
    """Verify defined layers input-output are valid before creation.

    Args:
        input_size: Input layer size
        architecture: Network architecture configuration

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

        if layer_type in ["embed", "conv2d", "vgg2l"]:
            if "output_size" not in layer_conf:
                raise ValueError(
                    "Layer " + str(i + 1) + ": Input layer format is: "
                    "{'layer_type: embed|conv2d|vgg2l', 'output_size': int, [...]}"
                )
            check_io.append((input_size, layer_conf["output_size"]))
        elif layer_type in ["transformer", "conformer"]:
            if not {"output_size", "linear_units", "attention_heads"}.issubset(
                layer_conf
            ):
                raise ValueError(
                    "Layer " + str(i + 1) + ": Transformer layer format is: "
                    "{'layer_type': transformer|conformer, 'output_size': int, "
                    "'linear_units': int, 'attention_heads': int, [...]}"
                )
            check_io.append((layer_conf["output_size"], layer_conf["output_size"]))
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


def get_positionwise_classes(
    positionwise_layer_type: str,
    positionwise_activation_type: str,
) -> torch.nn.Module:
    """Get positionwise layer related classes.

    Args:
        positionwise_layer_type: Type of positionwise layer
        positionwise_activation_type: Type of positiontwise layer activation

    Return:
        positionwise_layer: Positionwise layer class
        positionwise_activation: Positionwise activation class

    """
    if positionwise_layer_type == "linear":
        positionwise_layer = PositionwiseFeedForward
    elif positionwise_layer_type == "conv1d":
        positionwise_layer = MultiLayeredConv1d
    elif positionwise_layer_type == "conv1d-linear":
        positionwise_layer = Conv1dLinear
    else:
        raise NotImplementedError(
            "'positionwise_layer_type' argument only support 'linear' yet."
        )

    positionwise_activation = get_activation(positionwise_activation_type)

    return positionwise_layer, positionwise_activation


def get_positional_encoding_class(positional_encoding_type: str) -> torch.nn.Module:
    """Get positional encoding class.

    Args:
        positional_encoding_type: Type of positional encoding

    Returns:
        positional_encoding_class: Positional encoding class

    """
    if positional_encoding_type == "abs_pos":
        positional_encoding_class = PositionalEncoding
    elif positional_encoding_type == "scaled_abs_pos":
        positional_encoding_class = ScaledPositionalEncoding
    elif positional_encoding_type == "rel_pos":
        positional_encoding_class = RelPositionalEncoding
    else:
        raise NotImplementedError(
            "Invalid positional_encoding_type" + positional_encoding_type
        )

    return positional_encoding_class


def get_self_attention_class(self_attention_type: str) -> torch.nn.Module:
    """Get self-attention class.

    Args:
        self_attention_type: Type of self-attention

    Returns:
        self_attention_class: Self-attention class

    """
    if self_attention_type == "self_attn":
        self_attn_class = MultiHeadedAttention
    elif self_attention_type == "rel_selfattn":
        self_attn_class = RelPositionMultiHeadedAttention
    else:
        raise NotImplementedError(
            "self_attn_type should be either 'self_attn' or 'rel_self_attn'"
        )

    return self_attn_class


def create_transformer_layer(
    output_size: int,
    linear_units: int,
    attention_heads: int,
    positionwise_layer_type: str = "linear",
    positionwise_activation_type: str = "relu",
    dropout_rate: float = 0.0,
    positionwise_dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
) -> torch.nn.Module:
    """Create transformer layer.

    Args:
        output_size:
        linear_units:
        attention_heads:
        positionwise_layer_type:
        positionwise_activation_type:
        dropout_rate:
        positionwise_dropout_rate:
        attention_dropout_rate:
    """
    pw_layer, pw_activation = get_positionwise_classes(
        positionwise_layer_type, positionwise_activation_type
    )

    if positionwise_layer_type == "linear":
        pw_layer_args = (
            output_size,
            linear_units,
            positionwise_dropout_rate,
            pw_activation,
        )
    else:
        raise ValueError(
            "Transformer layer: positionwise_layer_type should be 'linear'."
        )

    return lambda: TransformerEncoderLayer(
        output_size,
        MultiHeadedAttention(attention_heads, output_size, attention_dropout_rate),
        pw_layer(*pw_layer_args),
        dropout_rate,
    )


def create_conformer_layer(
    output_size: int,
    linear_units: int,
    attention_heads: int,
    self_attention_layer_type: str = "rel_selfattn",
    positionwise_layer_type: str = "linear",
    positionwise_activation_type: str = "swish",
    positionwise_convolution_kernel_size: int = 3,
    positional_encoding_type: str = "rel_pos",
    positional_encoding_activation_type: str = "swish",
    macaron_style: bool = False,
    use_cnn_module: bool = True,
    cnn_module_kernel: int = 31,
    dropout_rate: float = 0.0,
    positionwise_dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
) -> torch.nn.Module:
    """Create transformer layer.

    Args:
        output_size:
        linear_units:
        attention_heads:
        positionwise_layer_type:
        positionwise_activation_type:
        dropout_rate:
        positionwise_dropout_rate:
        attention_dropout_rate:
    """
    pw_layer, pw_activation = get_positionwise_classes(
        positionwise_layer_type, positionwise_activation_type
    )

    if positionwise_layer_type == "linear":
        pw_layer_args = (
            output_size,
            linear_units,
            positionwise_dropout_rate,
            pw_activation,
        )
    else:
        pw_layer_args = (
            output_size,
            linear_units,
            positionwise_convolution_kernel_size,
            pw_activation,
        )

    self_att_layer = get_self_attention_class(self_attention_layer_type)
    self_att_layer_args = (
        attention_heads,
        output_size,
        attention_dropout_rate,
    )

    conv_layer = ConvolutionModule
    conv_layer_args = (output_size, cnn_module_kernel, pw_activation)

    return lambda: ConformerEncoderLayer(
        output_size,
        self_att_layer(*self_att_layer_args),
        pw_layer(*pw_layer_args),
        pw_layer(*pw_layer_args) if macaron_style else None,
        conv_layer(*conv_layer_args) if use_cnn_module else None,
        dropout_rate,
    )


def create_input_layer(
    input_size: int,
    output_size: int = 320,
    layer_type: str = "embed",
    positional_encoding_type: str = "abs_pos",
    dropout_rate: float = 0.0,
    positional_encoding_dropout_rate: float = 0.0,
    padding_idx: int = -1,
) -> torch.nn.Module:
    """Create input layer.

    Args:
        input_size: Input dimension
        output_size: Output dimension
        layer_type: Type of layer
        dropout_rate: Dropout rate
        position_encoding_type: Positional encoding layer type
        positional_encoding_dropout_rate: Positional encoding dropout rate

    Return:
        input_layer: Input layer

    """
    pos_enc_class = get_positional_encoding_class(positional_encoding_type)

    if layer_type == "linear":
        embed = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.LayerNorm(output_size),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            pos_enc_class(output_size, positional_encoding_dropout_rate),
        )
    elif layer_type == "conv2d":
        embed = Conv2dSubsampling(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_encoding_dropout_rate),
        )
    elif layer_type == "vgg2l":
        embed = VGG2L(input_size, output_size)
    elif layer_type == "embed":
        embed = torch.nn.Sequential(
            torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
            pos_enc_class(output_size, positional_encoding_dropout_rate),
        )
    elif layer_type is None:
        embed = torch.nn.Sequential(
            pos_enc_class(output_size, positional_encoding_dropout_rate)
        )
    else:
        raise NotImplementedError(
            "Supported input layer type: linear, conv2d, vgg2l and embed."
        )

    return embed


def create_network(
    architecture_type: str,
    input_size: int,
    architecture: List[Dict[str, Any]],
    repeat: int = 0,
    padding_idx: int = -1,
) -> Union[torch.nn.Module, MultiSequential, int]:
    """Create network based on architecture specification.

    Args:
        input_size: Input dimension
        architecture_part: Either 'encoder' or 'decoder'
        architecture: Network architecture configuration
        repeat: Number of times specified architecture should be repeated
        padding_idx: Index for embedding padding

    """
    assert architecture_type == "encoder", f'{"Only encoder is supported yet."}'

    output_size = verify_layers_io(input_size, architecture)

    input_layer = create_input_layer(
        input_size, padding_idx=padding_idx, **architecture[0]
    )

    architecture_modules = []
    for layer_conf in architecture[1:]:
        layer_type = layer_conf["layer_type"]
        layer_args = {x: layer_conf[x] for x in layer_conf if x != "layer_type"}

        if layer_type == "transformer":
            module = create_transformer_layer(**layer_args)
        elif layer_type == "conformer":
            module = create_conformer_layer(**layer_args)

        architecture_modules.append(module)

    if repeat > 1:
        architecture_modules = architecture_modules * repeat

    return (
        input_layer,
        MultiSequential(*[mod() for mod in architecture_modules]),
        output_size,
    )
