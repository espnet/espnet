"""Set of utilities methods to build customizable encoder/decoder."""

import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)


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
            "Invalid positional_encoding_type: " + positional_encoding_type
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
