"""Set of utilities methods to build customizable encoder/decoder."""

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)


def config_verification(
    supported_layers: Dict[str, Dict[str, Union[Tuple[str, str], str]]],
    architecture: List[Dict[str, Any]],
    repeat: int,
):
    """Verify specified layers type + input/output.

    If the specified architecture is valid, return last layer output dimension

    Args:
        supported_layers: Set of layers types and expected parameters
        architecture: Architecture configuration
        repeat: Number of times specified architecture (minus input layer)
                should be repeated

    Returns:
        (): last layer output dimension

    """

    def layer_verification(conf, part, i):
        if "layer_type" in conf:
            layer_type = conf["layer_type"]
        else:
            raise ValueError("layer_type is not defined in the " + str(i) + "th layer.")

        if layer_type in supported_layers[part]:
            params = supported_layers[part][layer_type]

            set_params = {params} if isinstance(params, str) else set(params)

            if not set_params.issubset(conf):
                raise ValueError(
                    "Layer "
                    + str(i)
                    + ", "
                    + layer_type
                    + ": Mandatory parameters are: "
                    + str(set_params)
                )

            _io = (
                (conf[params[0]], conf[params[1]])
                if isinstance(params, tuple)
                else (conf[params], conf[params])
            )

        else:
            raise NotImplementedError(
                "Layer "
                + str(i)
                + ": layer_type "
                + layer_type
                + " is not supported. "
                + "Expected: "
                + ", ".join(supported_layers[part].keys())
            )

        return _io

    assert check_argument_types

    if repeat > 1:
        body = architecture[1:] * repeat
    else:
        body = architecture[1:]

    check_io = [layer_verification(architecture[0], "input", 1)]
    for i, layer_conf in enumerate(body):
        check_io.append(layer_verification(layer_conf, "body", (i + 2)))

    for i in range(1, len(check_io)):
        if check_io[(i - 1)][1] != check_io[i][0]:
            raise ValueError(
                "Output-Input mismatch between layers " + str(i) + " and " + str(i + 1)
            )

    return check_io[-1][1]


def get_positionwise_class(positionwise_type: str) -> torch.nn.Module:
    """Get positionwise layer class.

    Args:
        positionwise_layer_type: Type of positionwise layer

    Return:
        positionwise_layer: Positionwise layer class

    """
    if positionwise_type == "linear":
        positionwise_class = PositionwiseFeedForward
    elif positionwise_type == "conv1d":
        positionwise_class = MultiLayeredConv1d
    elif positionwise_type == "conv1d-linear":
        positionwise_class = Conv1dLinear
    else:
        raise NotImplementedError(
            "positionwise_layer_type only supports 'conv1d', "
            "'conv1d-linear' and 'linear'. Specified: " + positionwise_type
        )

    return positionwise_class


def get_positional_encoding_class(
    positional_encoding_type: str,
    self_attention_type: str,
) -> torch.nn.Module:
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
        assert self_attention_type == "rel_self_attn"

        positional_encoding_class = RelPositionalEncoding
    else:
        raise NotImplementedError(
            "positional_encoding_type only supports': 'abs_pos', "
            "'scaled_pos' and 'rel_pos'. Specified: " + positional_encoding_type
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
        self_attention_class = MultiHeadedAttention
    elif self_attention_type == "rel_self_attn":
        self_attention_class = RelPositionMultiHeadedAttention
    else:
        raise NotImplementedError(
            "self_attention_type only supports': 'abs_pos', "
            "'scaled_pos' and 'rel_pos'. Specified: " + self_attention_type
        )

    return self_attention_class


def get_lightweight_dynamic_convolution_class(layer_type: str) -> torch.nn.Module:
    """Get lightweight/dynamic convolution class.

    Args:
        layer_type: Type of lightweight/dynamic layer

    Returns:
        ld_class: Lightweight/Dynamic class

    """
    if layer_type == "lightweight_conv":
        ld_class = LightweightConvolution
    elif layer_type == "dynamic_conv":
        ld_class = DynamicConvolution
    elif layer_type == "lightweight_conv2d":
        ld_class = LightweightConvolution2D
    elif layer_type == "dynamic_conv2d":
        ld_class = DynamicConvolution2D

    return ld_class
