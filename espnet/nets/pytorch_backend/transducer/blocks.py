"""Set of methods to create transformer-based block."""

import torch

from espnet.nets.pytorch_backend.transducer.causal_conv1d import CausalConv1d
from espnet.nets.pytorch_backend.transducer.tdnn import TDNN
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


def build_input_layer(
    input_layer,
    idim,
    odim,
    pos_enc_class,
    dropout_rate_embed,
    dropout_rate,
    pos_dropout_rate,
    padding_idx,
):
    """Build input layer.

    Args:
        input_layer (str): input layer type
        idim (int): input dimension
        odim (int): output dimension
        pos_enc_class (class): PositionalEncoding or ScaledPositionalEncoding
        dropout_rate_embed (float): dropout rate for embedding
        dropout_rate (float): dropout rate
        pos_dropout_rate (float): dropout rate for positional encoding
        padding_idx (int): padding index for embedding

    Returns:
        (*): input layer

    """
    if input_layer == "linear":
        return torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            pos_enc_class(odim, pos_dropout_rate),
        )
    elif input_layer == "conv2d":
        return Conv2dSubsampling(idim, odim, dropout_rate)
    elif input_layer == "vgg2l":
        return VGG2L(idim, odim)
    elif input_layer == "embed":
        return torch.nn.Sequential(
            torch.nn.Embedding(idim, odim, padding_idx=padding_idx),
            pos_enc_class(odim, pos_dropout_rate),
        )
    elif input_layer == "t-linear":
        return torch.nn.Linear(idim, odim)
    elif input_layer == "c-embed":
        return torch.nn.Sequential(
            torch.nn.Embedding(idim, odim, padding_idx=padding_idx),
            torch.nn.Dropout(dropout_rate_embed),
        )
    elif input_layer is None:
        return pos_enc_class(odim, pos_dropout_rate)
    else:
        raise NotImplementedError("Support: linear, conv2d, vgg2l and embed")


def build_transformer_layer(
    transformer_layer_class,
    block_arch,
    pw_layer_type,
    pw_conv_kernel_size,
    pos_dropout_rate,
    att_dropout_rate,
    dropout_rate,
):
    """Build function for Transformer layer.

    Args:
        transformer_layer_class (class): whether EncoderLayer or DecoderLayer
        block_arch (dict): layer main arguments
        pos_layer_type (str): positionwise layer type
        pos_conv_kernel_size (int) : kernel size for positionwise conv1d layer
        pos_dropout_rate (float): dropout rate for positional encoding
        att_dropout_rate (float): dropout rate for attention
        dropout_rate (float): dropout rate

    Returns:
        (function): function to create Transformer layer

    """
    if {"d_hidden", "d_ff", "heads"} > block_arch.keys():
        raise ValueError(
            "Transformer layer format is: {'type: transformer', "
            "'d_hidden': int, 'd_ff': int, 'heads': int}"
        )

    d_hidden = block_arch["d_hidden"]
    d_ff = block_arch["d_ff"]
    heads = block_arch["heads"]

    if pw_layer_type == "linear":
        pw_layer = PositionwiseFeedForward
        pw_layer_args = (d_hidden, d_ff, dropout_rate)
    elif pw_layer_type == "conv1d":
        pw_layer = MultiLayeredConv1d
        pw_layer_args = (d_hidden, d_ff, pw_conv_kernel_size, dropout_rate)
    elif pw_layer_type == "conv1d-linear":
        pw_layer = Conv1dLinear
        pw_layer_args = (d_hidden, d_ff, pw_conv_kernel_size, dropout_rate)
    else:
        raise NotImplementedError("Support only linear or conv1d.")

    return lambda: transformer_layer_class(
        d_hidden,
        MultiHeadedAttention(heads, d_hidden, att_dropout_rate),
        pw_layer(*pw_layer_args),
        dropout_rate,
    )


def build_causal_conv1d_layer(block_arch):
    """Build function for CausalConv1d layer.

    Args:
        block_arch (dict): layer arguments

    Returns:
        (function): function to create CausalConv1d layer

    """
    if {"idim", "odim", "kernel_size"} > block_arch.keys():
        raise ValueError(
            "CausalConv1d layer format is: {'type: causal-conv1d', "
            "'idim': int, 'odim': int, 'kernel_size': int}"
        )

    idim = block_arch["idim"]
    odim = block_arch["odim"]
    kernel_size = block_arch["kernel_size"]

    return lambda: CausalConv1d(idim, odim, kernel_size)


def build_tdnn_layer(block_arch):
    """Build function for TDNN layer.

    Args:
        block_arch (dict): layer arguments

    Returns:
        (function): function to create TDNN layer

    """
    if {"idim", "odim", "ctx_size", "dilation", "stride"} > block_arch.keys():
        raise ValueError(
            "TDNN block format is: {'type: tdnn', "
            "'idim': int, 'odim': int, 'ctx_size': int, "
            "'dilation': int, 'stride': int}"
        )

    idim = block_arch["idim"]
    odim = block_arch["odim"]
    ctx_size = block_arch["ctx_size"]
    dilation = block_arch["dilation"]
    stride = block_arch["stride"]

    return lambda: TDNN(idim, odim, ctx_size, dilation, stride)


def build_blocks(
    idim,
    input_layer,
    block_arch,
    transformer_layer_class,
    repeat_block=0,
    pos_enc_class=PositionalEncoding,
    positionwise_layer_type="linear",
    positionwise_conv_kernel_size=1,
    dropout_rate_embed=0.0,
    dropout_rate=0.0,
    positional_dropout_rate=0.0,
    att_dropout_rate=0.0,
    padding_idx=-1,
):
    """Build block for transformer-based models.

    Args:
        idim (int): dimension of inputs
        input_layer (str): input layer type
        block_arch (list[dict]): list of layer definitions in block
        transformer_layer_class (class): whether EncoderLayer or DecoderLayer
        repeat_block (int): if N > 1, repeat block N times
        pos_enc_class (class): PositionalEncoding or ScaledPositionalEncoding
        positionwise_layer_type (str): linear of conv1d
        positionwise_conv_kernel_size (int) : kernel size of positionwise conv1d layer
        dropout_rate_embed (float): dropout rate
        dropout_rate (float): dropout rate
        positional_dropout_rate (float): dropout rate for positional encoding
        att_dropout_rate (float): dropout rate for attention
        padding_idx (int): padding index for embedding

    Returns:
        in_layer (*): input layer
        block (MultiSequential): block of layers created from specified config
        out_dim (int): dimension of block output

    """
    fn_modules = []

    for i in range(len(block_arch)):
        if "type" in block_arch[i]:
            layer_type = block_arch[i]["type"]
        else:
            raise ValueError("type is not defined in following block: ", block_arch[i])

        if layer_type == "tdnn":
            if i == 0:
                input_layer = "t-linear"

            module = build_tdnn_layer(block_arch[i])
        elif layer_type == "transformer":
            module = build_transformer_layer(
                transformer_layer_class,
                block_arch[i],
                positionwise_layer_type,
                positionwise_conv_kernel_size,
                positional_dropout_rate,
                att_dropout_rate,
                dropout_rate,
            )
        elif layer_type == "causal-conv1d":
            if i == 0:
                input_layer = "c-embed"

            module = build_causal_conv1d_layer(block_arch[i])
        else:
            raise NotImplementedError(
                "Transformer layer type currently supported: "
                "tdnn, causal-conv1d or transformer"
            )

        fn_modules.append(module)

    if repeat_block > 1:
        fn_modules = fn_modules * repeat_block

    if block_arch[0]["type"] in ("tdnn", "causal-conv1d"):
        input_layer_odim = block_arch[0]["idim"]
    else:
        input_layer_odim = block_arch[0]["d_hidden"]

    in_layer = build_input_layer(
        input_layer,
        idim,
        input_layer_odim,
        pos_enc_class,
        dropout_rate_embed,
        dropout_rate,
        positional_dropout_rate,
        padding_idx,
    )

    if block_arch[-1]["type"] in ("tdnn", "causal-conv1d"):
        out_dim = block_arch[-1]["idim"]
    else:
        out_dim = block_arch[-1]["d_hidden"]

    return in_layer, MultiSequential(*[fn() for fn in fn_modules]), out_dim
