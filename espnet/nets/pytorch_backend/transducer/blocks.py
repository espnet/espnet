"""Set of methods to create custom architecture."""

from collections import Counter

import torch

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import (
    EncoderLayer as ConformerEncoderLayer,  # noqa: H301
)

from espnet.nets.pytorch_backend.nets_utils import get_activation

from espnet.nets.pytorch_backend.transducer.causal_conv1d import CausalConv1d
from espnet.nets.pytorch_backend.transducer.transformer_decoder_layer import (
    DecoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.tdnn import TDNN
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


def check_and_prepare(net_part, blocks_arch, input_layer):
    """Check consecutive block shapes match and prepare input parameters.

    Args:
        net_part (str): either 'encoder' or 'decoder'
        blocks_arch (list): list of blocks for network part (type and parameters)
        input_layer (str): input layer type

    Return:
        input_layer (str): input layer type
        input_layer_odim (int): output dim of input layer
        input_dropout_rate (float): dropout rate of input layer
        input_pos_dropout_rate (float): dropout rate of input layer positional enc.
        out_dim (int): output dim of last block

    """
    input_dropout_rate = sorted(
        Counter(
            b["dropout-rate"] for b in blocks_arch if "dropout-rate" in b
        ).most_common(),
        key=lambda x: x[0],
        reverse=True,
    )

    input_pos_dropout_rate = sorted(
        Counter(
            b["pos-dropout-rate"] for b in blocks_arch if "pos-dropout-rate" in b
        ).most_common(),
        key=lambda x: x[0],
        reverse=True,
    )

    input_dropout_rate = input_dropout_rate[0][0] if input_dropout_rate else 0.0
    input_pos_dropout_rate = (
        input_pos_dropout_rate[0][0] if input_pos_dropout_rate else 0.0
    )

    cmp_io = []
    has_transformer = False
    has_conformer = False
    for i in range(len(blocks_arch)):
        if "type" in blocks_arch[i]:
            block_type = blocks_arch[i]["type"]
        else:
            raise ValueError("type is not defined in the " + str(i + 1) + "th block.")

        if block_type == "transformer":
            if not {"d_hidden", "d_ff", "heads"}.issubset(blocks_arch[i]):
                raise ValueError(
                    "Block "
                    + str(i + 1)
                    + "in "
                    + net_part
                    + ": Transformer block format is: {'type: transformer', "
                    "'d_hidden': int, 'd_ff': int, 'heads': int, [...]}"
                )

            has_transformer = True
            cmp_io.append((blocks_arch[i]["d_hidden"], blocks_arch[i]["d_hidden"]))
        elif block_type == "conformer":
            if net_part != "encoder":
                raise ValueError(
                    "Block " + str(i + 1) + ": conformer type is only for encoder part."
                )

            if not {
                "d_hidden",
                "d_ff",
                "heads",
                "macaron_style",
                "use_conv_mod",
            }.issubset(blocks_arch[i]):
                raise ValueError(
                    "Block "
                    + str(i + 1)
                    + " in "
                    + net_part
                    + ": Conformer block format is {'type: conformer', "
                    "'d_hidden': int, 'd_ff': int, 'heads': int, "
                    "'macaron_style': bool, 'use_conv_mod': bool, [...]}"
                )

            if (
                blocks_arch[i]["use_conv_mod"] is True
                and "conv_mod_kernel" not in blocks_arch[i]
            ):
                raise ValueError(
                    "Block "
                    + str(i + 1)
                    + ": 'use_conv_mod' is True but 'use_conv_kernel' is not specified"
                )

            has_conformer = True
            cmp_io.append((blocks_arch[i]["d_hidden"], blocks_arch[i]["d_hidden"]))
        elif block_type == "causal-conv1d":
            if not {"idim", "odim", "kernel_size"}.issubset(blocks_arch[i]):
                raise ValueError(
                    "Block "
                    + str(i + 1)
                    + " in "
                    + net_part
                    + ": causal conv1d block format is: {'type: causal-conv1d', "
                    "'idim': int, 'odim': int, 'kernel_size': int}"
                )

            if i == 0:
                input_layer = "c-embed"

            cmp_io.append((blocks_arch[i]["idim"], blocks_arch[i]["odim"]))
        elif block_type == "tdnn":
            if not {"idim", "odim", "ctx_size", "dilation", "stride"}.issubset(
                blocks_arch[i]
            ):
                raise ValueError(
                    "Block "
                    + str(i + 1)
                    + " in "
                    + net_part
                    + ": TDNN block format is: {'type: tdnn', "
                    "'idim': int, 'odim': int, 'ctx_size': int, "
                    "'dilation': int, 'stride': int, [...]}"
                )

            cmp_io.append((blocks_arch[i]["idim"], blocks_arch[i]["odim"]))
        else:
            raise NotImplementedError(
                "Wrong type for block "
                + str(i + 1)
                + " in "
                + net_part
                + ". Currently supported: "
                "tdnn, causal-conv1d or transformer"
            )

    if has_transformer and has_conformer:
        raise NotImplementedError(
            net_part + ": transformer and conformer blocks "
            "can't be defined in the same net part."
        )

    for i in range(1, len(cmp_io)):
        if cmp_io[(i - 1)][1] != cmp_io[i][0]:
            raise ValueError(
                "Output/Input mismatch between blocks "
                + str(i)
                + " and "
                + str(i + 1)
                + " in "
                + net_part
            )

    if blocks_arch[0]["type"] in ("tdnn", "causal-conv1d"):
        input_layer_odim = blocks_arch[0]["idim"]
    else:
        input_layer_odim = blocks_arch[0]["d_hidden"]

    if blocks_arch[-1]["type"] in ("tdnn", "causal-conv1d"):
        out_dim = blocks_arch[-1]["odim"]
    else:
        out_dim = blocks_arch[-1]["d_hidden"]

    return (
        input_layer,
        input_layer_odim,
        input_dropout_rate,
        input_pos_dropout_rate,
        out_dim,
    )


def get_pos_enc_and_att_class(net_part, pos_enc_type, self_attn_type):
    """Get positional encoding and self attention module class.

    Args:
        net_part (str): either 'encoder' or 'decoder'
        pos_enc_type (str): positional encoding type
        self_attn_type (str): self-attention type

    Return:
        pos_enc_class (torch.nn.Module): positional encoding class
        self_attn_class (torch.nn.Module): self-attention class

    """
    if pos_enc_type == "abs_pos":
        pos_enc_class = PositionalEncoding
    elif pos_enc_type == "scaled_abs_pos":
        pos_enc_class = ScaledPositionalEncoding
    elif pos_enc_type == "rel_pos":
        if net_part == "encoder" and self_attn_type != "rel_self_attn":
            raise ValueError("'rel_pos' is only compatible with 'rel_self_attn'")
        pos_enc_class = RelPositionalEncoding
    else:
        raise NotImplementedError(
            "pos_enc_type should be either 'abs_pos', 'scaled_abs_pos' or 'rel_pos'"
        )

    if self_attn_type == "rel_self_attn":
        self_attn_class = RelPositionMultiHeadedAttention
    else:
        self_attn_class = MultiHeadedAttention

    return pos_enc_class, self_attn_class


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
        pos_enc_class (class): positional encoding class
        dropout_rate_embed (float): dropout rate for embedding layer
        dropout_rate (float): dropout rate for input layer
        pos_dropout_rate (float): dropout rate for positional encoding
        padding_idx (int): padding index for embedding input layer (if specified)

    Returns:
        (torch.nn.*): input layer module
        subsampling_factor (int): subsampling factor

    """
    if pos_enc_class.__name__ == "RelPositionalEncoding":
        pos_enc_class_subsampling = pos_enc_class(odim, pos_dropout_rate)
    else:
        pos_enc_class_subsampling = None

    if input_layer == "linear":
        return (
            torch.nn.Sequential(
                torch.nn.Linear(idim, odim),
                torch.nn.LayerNorm(odim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(odim, pos_dropout_rate),
            ),
            1,
        )
    elif input_layer == "conv2d":
        return Conv2dSubsampling(idim, odim, dropout_rate, pos_enc_class_subsampling), 4
    elif input_layer == "vgg2l":
        return VGG2L(idim, odim, pos_enc_class_subsampling), 4
    elif input_layer == "embed":
        return (
            torch.nn.Sequential(
                torch.nn.Embedding(idim, odim, padding_idx=padding_idx),
                pos_enc_class(odim, pos_dropout_rate),
            ),
            1,
        )
    elif input_layer == "c-embed":
        return (
            torch.nn.Sequential(
                torch.nn.Embedding(idim, odim, padding_idx=padding_idx),
                torch.nn.Dropout(dropout_rate_embed),
            ),
            1,
        )
    else:
        raise NotImplementedError("Support: linear, conv2d, vgg2l and embed")


def build_transformer_block(net_part, block_arch, pw_layer_type, pw_activation_type):
    """Build function for transformer block.

    Args:
        net_part (str): either 'encoder' or 'decoder'
        block_arch (dict): transformer block parameters
        pw_layer_type (str): positionwise layer type
        pw_activation_type (str): positionwise activation type

    Returns:
        (function): function to create transformer block

    """
    d_hidden = block_arch["d_hidden"]
    d_ff = block_arch["d_ff"]
    heads = block_arch["heads"]

    dropout_rate = block_arch["dropout-rate"] if "dropout-rate" in block_arch else 0.0
    pos_dropout_rate = (
        block_arch["pos-dropout-rate"] if "pos-dropout-rate" in block_arch else 0.0
    )
    att_dropout_rate = (
        block_arch["att-dropout-rate"] if "att-dropout-rate" in block_arch else 0.0
    )

    if pw_layer_type == "linear":
        pw_layer = PositionwiseFeedForward
        pw_activation = get_activation(pw_activation_type)
        pw_layer_args = (d_hidden, d_ff, pos_dropout_rate, pw_activation)
    else:
        raise NotImplementedError("Transformer block only supports linear yet.")

    if net_part == "encoder":
        transformer_layer_class = EncoderLayer
    elif net_part == "decoder":
        transformer_layer_class = DecoderLayer

    return lambda: transformer_layer_class(
        d_hidden,
        MultiHeadedAttention(heads, d_hidden, att_dropout_rate),
        pw_layer(*pw_layer_args),
        dropout_rate,
    )


def build_conformer_block(
    block_arch,
    self_attn_class,
    pw_layer_type,
    pw_activation_type,
    conv_mod_activation_type,
):
    """Build function for conformer block.

    Args:
        block_arch (dict): conformer block parameters
        self_attn_type (str): self-attention module type
        pw_layer_type (str): positionwise layer type
        pw_activation_type (str): positionwise activation type
        conv_mod_activation_type (str): convolutional module activation type

    Returns:
        (function): function to create conformer block

    """
    d_hidden = block_arch["d_hidden"]
    d_ff = block_arch["d_ff"]
    heads = block_arch["heads"]
    macaron_style = block_arch["macaron_style"]
    use_conv_mod = block_arch["use_conv_mod"]

    dropout_rate = block_arch["dropout-rate"] if "dropout-rate" in block_arch else 0.0
    pos_dropout_rate = (
        block_arch["pos-dropout-rate"] if "pos-dropout-rate" in block_arch else 0.0
    )
    att_dropout_rate = (
        block_arch["att-dropout-rate"] if "att-dropout-rate" in block_arch else 0.0
    )

    if pw_layer_type == "linear":
        pw_layer = PositionwiseFeedForward
        pw_activation = get_activation(pw_activation_type)
        pw_layer_args = (d_hidden, d_ff, pos_dropout_rate, pw_activation)
    else:
        raise NotImplementedError("Conformer block only supports linear yet.")

    if use_conv_mod:
        conv_layer = ConvolutionModule
        conv_activation = get_activation(conv_mod_activation_type)
        conv_layers_args = (d_hidden, block_arch["conv_mod_kernel"], conv_activation)

    return lambda: ConformerEncoderLayer(
        d_hidden,
        self_attn_class(heads, d_hidden, att_dropout_rate),
        pw_layer(*pw_layer_args),
        pw_layer(*pw_layer_args) if macaron_style else None,
        conv_layer(*conv_layers_args) if use_conv_mod else None,
        dropout_rate,
    )


def build_causal_conv1d_block(block_arch):
    """Build function for causal conv1d block.

    Args:
        block_arch (dict): causal conv1d block parameters

    Returns:
        (function): function to create causal conv1d block

    """
    idim = block_arch["idim"]
    odim = block_arch["odim"]
    kernel_size = block_arch["kernel_size"]

    return lambda: CausalConv1d(idim, odim, kernel_size)


def build_tdnn_block(block_arch):
    """Build function for tdnn block.

    Args:
        block_arch (dict): tdnn block parameters

    Returns:
        (function): function to create tdnn block

    """
    idim = block_arch["idim"]
    odim = block_arch["odim"]
    ctx_size = block_arch["ctx_size"]
    dilation = block_arch["dilation"]
    stride = block_arch["stride"]

    use_batch_norm = (
        block_arch["use-batch-norm"] if "use-batch-norm" in block_arch else False
    )
    use_relu = block_arch["use-relu"] if "use-relu" in block_arch else False

    dropout_rate = block_arch["dropout-rate"] if "dropout-rate" in block_arch else 0.0

    return lambda: TDNN(
        idim,
        odim,
        ctx_size=ctx_size,
        dilation=dilation,
        stride=stride,
        dropout_rate=dropout_rate,
        batch_norm=use_batch_norm,
        relu=use_relu,
    )


def build_blocks(
    net_part,
    idim,
    input_layer,
    blocks_arch,
    repeat_block=0,
    self_attn_type="self_attn",
    positional_encoding_type="abs_pos",
    positionwise_layer_type="linear",
    positionwise_activation_type="relu",
    conv_mod_activation_type="relu",
    dropout_rate_embed=0.0,
    padding_idx=-1,
):
    """Build block for customizable architecture.

    Args:
        net_part (str): either 'encoder' or 'decoder'
        idim (int): dimension of inputs
        input_layer (str): input layer type
        blocks_arch (list): list of blocks for network part (type and parameters)
        repeat_block (int): repeat provided blocks N times if N > 1
        positional_encoding_type (str): positional encoding layer type
        positionwise_layer_type (str): linear
        positionwise_activation_type (str): positionwise activation type
        conv_mod_activation_type (str): convolutional module activation type
        dropout_rate_embed (float): dropout rate for embedding
        padding_idx (int): padding index for embedding input layer (if specified)

    Returns:
        in_layer (torch.nn.*): input layer
        all_blocks (MultiSequential): all blocks for network part
        out_dim (int): dimension of last block output
        conv_subsampling_factor (int): subsampling factor in frontend CNN

    """
    fn_modules = []

    (
        input_layer,
        input_layer_odim,
        input_dropout_rate,
        input_pos_dropout_rate,
        out_dim,
    ) = check_and_prepare(net_part, blocks_arch, input_layer)

    pos_enc_class, self_attn_class = get_pos_enc_and_att_class(
        net_part, positional_encoding_type, self_attn_type
    )

    in_layer, conv_subsampling_factor = build_input_layer(
        input_layer,
        idim,
        input_layer_odim,
        pos_enc_class,
        dropout_rate_embed,
        input_dropout_rate,
        input_pos_dropout_rate,
        padding_idx,
    )

    for i in range(len(blocks_arch)):
        block_type = blocks_arch[i]["type"]

        if block_type == "tdnn":
            module = build_tdnn_block(blocks_arch[i])
        elif block_type == "transformer":
            module = build_transformer_block(
                net_part,
                blocks_arch[i],
                positionwise_layer_type,
                positionwise_activation_type,
            )
        elif block_type == "conformer":
            module = build_conformer_block(
                blocks_arch[i],
                self_attn_class,
                positionwise_layer_type,
                positionwise_activation_type,
                conv_mod_activation_type,
            )
        elif block_type == "causal-conv1d":
            module = build_causal_conv1d_block(blocks_arch[i])

        fn_modules.append(module)

    if repeat_block > 1:
        fn_modules = fn_modules * repeat_block

    return (
        in_layer,
        MultiSequential(*[fn() for fn in fn_modules]),
        out_dim,
        conv_subsampling_factor,
    )
