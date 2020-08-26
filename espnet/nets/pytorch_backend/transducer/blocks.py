"""Set of methods to create transformer-based block."""

from collections import Counter

import torch

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import (
    EncoderLayer as ConformerEncoderLayer,  # noqa: H301
)

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


def check_and_prepare(block_part, all_blocks, input_layer):
    """Check consecutive block shapes match and prepare input parameters.

    Args:
        block_part (str): either 'encoder' or 'decoder'
        all_blocks (list): all blocks main arguments
        input_layer (str): input layer type

    Return:
        input_layer (str): input layer type
        input_layer_odim (int): output dim of input layer
        input_dropout_rate (float): dropout rate of input layer
        input_pos_dropout_rate (float): dropout rate of pos. enc. in input layer
        out_dim (int): output dim of last block

    """
    if all_blocks[0]["type"] in ("tdnn", "causal-conv1d"):
        input_layer_odim = all_blocks[0]["idim"]
    else:
        input_layer_odim = all_blocks[0]["d_hidden"]

    input_dropout_rate = sorted(
        Counter(
            b["dropout-rate"] for b in all_blocks if "dropout-rate" in b
        ).most_common(),
        key=lambda x: x[0],
        reverse=True,
    )

    input_pos_dropout_rate = sorted(
        Counter(
            b["pos-dropout-rate"] for b in all_blocks if "pos-dropout-rate" in b
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
    for i in range(len(all_blocks)):
        if "type" in all_blocks[i]:
            layer_type = all_blocks[i]["type"]
        else:
            raise ValueError("type is not defined in the " + str(i) + "th block.")

        if layer_type == "transformer":
            if not {"d_hidden", "d_ff", "heads"}.issubset(all_blocks[i]):
                raise ValueError(
                    "Block "
                    + str(i)
                    + "in "
                    + block_part
                    + ": Transformer layer format is: {'type: transformer', "
                    "'d_hidden': int, 'd_ff': int, 'heads': int, [...]}"
                )

            has_transformer = True
            cmp_io.append((all_blocks[i]["d_hidden"], all_blocks[i]["d_hidden"]))
        elif layer_type == "conformer":
            if block_part != "encoder":
                raise ValueError(
                    "Block " + str(i) + ": conformer type is only for encoder part."
                )

            if not {
                "d_hidden",
                "d_ff",
                "heads",
                "macaron_style",
                "use_conv_mod",
            }.issubset(all_blocks[i]):
                raise ValueError(
                    "Block "
                    + str(i)
                    + " in "
                    + block_part
                    + ": Conformer layer format is {'type: conformer', "
                    "'d_hidden': int, 'd_ff': int, 'heads': int, "
                    "'macaron_style': bool, 'use_conv_mod': bool, [...]}"
                )

            if (
                all_blocks[i]["use_conv_mod"] is True
                and "conv_mod_kernel" not in all_blocks[i]
            ):
                raise ValueError(
                    "Block "
                    + str(i)
                    + ": 'use_conv_mod' is True but 'use_conv_kernel' is not specified"
                )

            if i == 0 and input_layer == "conv2d":
                input_layer = "conformer-conv2d"

            has_conformer = True
            cmp_io.append((all_blocks[i]["d_hidden"], all_blocks[i]["d_hidden"]))
        elif layer_type == "causal-conv1d":
            if not {"idim", "odim", "kernel_size"}.issubset(all_blocks[i]):
                raise ValueError(
                    "Block "
                    + str(i)
                    + " in "
                    + block_part
                    + ": CausalConv1d layer format is: {'type: causal-conv1d', "
                    "'idim': int, 'odim': int, 'kernel_size': int}"
                )

            if i == 0:
                input_layer = "c-embed"

            cmp_io.append((all_blocks[i]["idim"], all_blocks[i]["odim"]))
        elif layer_type == "tdnn":
            if not {"idim", "odim", "ctx_size", "dilation", "stride"}.issubset(
                all_blocks[i]
            ):
                raise ValueError(
                    "Block "
                    + str(i)
                    + " in "
                    + block_part
                    + ": TDNN block format is: {'type: tdnn', "
                    "'idim': int, 'odim': int, 'ctx_size': int, "
                    "'dilation': int, 'stride': int, [...]}"
                )

            if i == 0:
                input_layer = "t-linear"

            cmp_io.append((all_blocks[i]["idim"], all_blocks[i]["odim"]))
        else:
            raise NotImplementedError(
                "Wrong type for block "
                + str(i)
                + " in "
                + block_part
                + ". Currently supported: "
                "tdnn, causal-conv1d or transformer"
            )

    if has_transformer and has_conformer:
        raise NotImplementedError(
            block_part
            + ": transformer and conformer layers can't be defined in the same block."
        )

    for i in range(1, len(cmp_io)):
        if cmp_io[(i - 1)][1] != cmp_io[i][0]:
            raise ValueError(
                "Output/Input mismatch between block "
                + str(i - 1)
                + " and "
                + str(i)
                + " in "
                + block_part
            )

    if all_blocks[-1]["type"] in ("tdnn", "causal-conv1d"):
        out_dim = all_blocks[-1]["idim"]
    else:
        out_dim = all_blocks[-1]["d_hidden"]

    return (
        input_layer,
        input_layer_odim,
        input_dropout_rate,
        input_pos_dropout_rate,
        out_dim,
    )


def get_pos_enc_and_att_class(block_part, pos_enc_type, self_attn_type):
    """Get positional encoding and self attention module class.

    Args:
        block_part (str): either 'encoder' or 'decoder'
        pos_enc_type (str): positional encoding type
        self_attn_type (str): self attention type

    Return:
        pos_enc_class (torch.nn.Module): positional encoding class
        self_attn_class (torch.nn.Module): self attention class

    """
    if pos_enc_type == "abs_pos":
        pos_enc_class = PositionalEncoding
    elif pos_enc_type == "scaled_abs_pos":
        pos_enc_class = ScaledPositionalEncoding
    elif pos_enc_type == "rel_pos":
        if block_part == "encoder" and self_attn_type != "rel_self_attn":
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
        pos_enc_class (class):
            PositionalEncoding, ScaledPositionalEncoding or RelPositionalEncoding
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
    elif input_layer == "conformer-conv2d":
        return Conv2dSubsampling(
            idim, odim, dropout_rate, pos_enc_class(odim, pos_dropout_rate)
        )
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


def build_transformer_layer(block_part, block_arch, pw_layer_type):
    """Build function for Transformer layer.

    Args:
        transformer_layer_class (class): whether EncoderLayer or DecoderLayer
        block_arch (dict): layer main arguments
        pw_layer_type (str): positionwise layer type

    Returns:
        (function): function to create Transformer layer

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
        pw_layer_args = (d_hidden, d_ff, pos_dropout_rate)
    else:
        raise NotImplementedError("Transformer block only supports linear yet.")

    if block_part == "encoder":
        transformer_layer_class = EncoderLayer
    elif block_part == "decoder":
        transformer_layer_class = DecoderLayer

    return lambda: transformer_layer_class(
        d_hidden,
        MultiHeadedAttention(heads, d_hidden, att_dropout_rate),
        pw_layer(*pw_layer_args),
        dropout_rate,
    )


def build_conformer_layer(
    block_part, block_arch, pw_layer_type, self_attn_class, pos_enc_class
):
    """Build function for conformer layer.

    Args:

        block_arch (dict): layer main arguments
        pw_layer_type (str): positionwise layer type
        self_attn_type (str): self-attention module type
        pos_enc_class (str): positional encoding class

    Returns:
        (function): function to create Transformer layer

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
        pw_layer_args = (d_hidden, d_ff, pos_dropout_rate)
    else:
        raise NotImplementedError("Conformer block only supports linear yet.")

    if use_conv_mod:
        conv_layer = ConvolutionModule
        conv_layers_args = (d_hidden, block_arch["conv_mod_kernel"])

    return lambda: ConformerEncoderLayer(
        d_hidden,
        self_attn_class(heads, d_hidden, att_dropout_rate),
        pw_layer(*pw_layer_args),
        pw_layer(*pw_layer_args) if macaron_style else None,
        conv_layer(*conv_layers_args) if use_conv_mod else None,
        dropout_rate,
    )


def build_causal_conv1d_layer(block_arch):
    """Build function for CausalConv1d layer.

    Args:
        block_arch (dict): layer arguments

    Returns:
        (function): function to create CausalConv1d layer

    """
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
    idim = block_arch["idim"]
    odim = block_arch["odim"]
    ctx_size = block_arch["ctx_size"]
    dilation = block_arch["dilation"]
    stride = block_arch["stride"]

    dropout_rate = block_arch["dropout-rate"] if "dropout-rate" in block_arch else 0.0

    return lambda: TDNN(
        idim,
        odim,
        ctx_size=ctx_size,
        dilation=dilation,
        stride=stride,
        dropout_rate=dropout_rate,
    )


def build_blocks(
    block_part,
    idim,
    input_layer,
    block_arch,
    repeat_block=0,
    self_attn_type="self_attn",
    positional_encoding_type="abs_pos",
    positionwise_layer_type="linear",
    dropout_rate_embed=0.0,
    padding_idx=-1,
):
    """Build block for transformer-based models.

    Args:
        block_part (class): either 'encoder' or 'decoder'
        idim (int): dimension of inputs
        input_layer (str): input layer type
        block_arch (list[dict]): list of layer definitions in block
        repeat_block (int): if N > 1, repeat block N times
        positional_encoding_type (str): positional encoding layer type
        positionwise_layer_type (str): linear
        dropout_rate_embed (float): dropout rate for embedding
        padding_idx (int): padding index for embedding

    Returns:
        in_layer (*): input layer
        block (MultiSequential): block of layers created from specified config
        out_dim (int): dimension of block output

    """
    fn_modules = []

    (
        input_layer,
        input_layer_odim,
        input_dropout_rate,
        input_pos_dropout_rate,
        out_dim,
    ) = check_and_prepare(block_part, block_arch, input_layer)

    pos_enc_class, self_attn_class = get_pos_enc_and_att_class(
        block_part, positional_encoding_type, self_attn_type
    )

    in_layer = build_input_layer(
        input_layer,
        idim,
        input_layer_odim,
        pos_enc_class,
        dropout_rate_embed,
        input_dropout_rate,
        input_pos_dropout_rate,
        padding_idx,
    )

    for i in range(len(block_arch)):
        layer_type = block_arch[i]["type"]

        if layer_type == "tdnn":
            module = build_tdnn_layer(block_arch[i])
        elif layer_type == "transformer":
            module = build_transformer_layer(
                block_part,
                block_arch[i],
                positionwise_layer_type,
            )
        elif layer_type == "conformer":
            module = build_conformer_layer(
                block_part,
                block_arch[i],
                positionwise_layer_type,
                self_attn_class,
                pos_enc_class,
            )
        elif layer_type == "causal-conv1d":
            module = build_causal_conv1d_layer(block_arch[i])

        fn_modules.append(module)

    if repeat_block > 1:
        fn_modules = fn_modules * repeat_block

    return in_layer, MultiSequential(*[fn() for fn in fn_modules]), out_dim
