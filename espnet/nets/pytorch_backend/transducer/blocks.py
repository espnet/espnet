"""Set of methods to create custom architecture."""

from typing import Any, Dict, List, Tuple, Union

import torch

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import (
    EncoderLayer as ConformerEncoderLayer,
)
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transducer.conv1d_nets import CausalConv1d, Conv1d
from espnet.nets.pytorch_backend.transducer.transformer_decoder_layer import (
    TransformerDecoderLayer,
)
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


def verify_block_arguments(
    net_part: str,
    block: Dict[str, Any],
    num_block: int,
) -> Tuple[int, int]:
    """Verify block arguments are valid.

    Args:
        net_part: Network part, either 'encoder' or 'decoder'.
        block: Block parameters.
        num_block: Block ID.

    Return:
        block_io: Input and output dimension of the block.

    """
    block_type = block.get("type")

    if block_type is None:
        raise ValueError(
            "Block %d in %s doesn't a type assigned.", (num_block, net_part)
        )

    if block_type == "transformer":
        arguments = {"d_hidden", "d_ff", "heads"}
    elif block_type == "conformer":
        arguments = {
            "d_hidden",
            "d_ff",
            "heads",
            "macaron_style",
            "use_conv_mod",
        }

        if net_part == "decoder":
            raise ValueError("Decoder does not support 'conformer'.")

        if block.get("use_conv_mod", None) is True and "conv_mod_kernel" not in block:
            raise ValueError(
                "Block %d: 'use_conv_mod' is True but "
                " 'conv_mod_kernel' is not specified" % num_block
            )
    elif block_type == "causal-conv1d":
        arguments = {"idim", "odim", "kernel_size"}

        if net_part == "encoder":
            raise ValueError("Encoder does not support 'causal-conv1d'.")

    elif block_type == "conv1d":
        arguments = {"idim", "odim", "kernel_size"}

        if net_part == "decoder":
            raise ValueError("Decoder does not support 'conv1d.'")
    else:
        raise NotImplementedError(
            "Wrong type. Currently supported: "
            "causal-conv1d, conformer, conv-nd or transformer."
        )

    if not arguments.issubset(block):
        raise ValueError(
            "%s in %s in position %d: Expected block arguments : %s."
            " See tutorial page for more information."
            % (block_type, net_part, num_block, arguments)
        )

    if block_type in ("transformer", "conformer"):
        block_io = (block["d_hidden"], block["d_hidden"])
    else:
        block_io = (block["idim"], block["odim"])

    return block_io


def prepare_input_layer(
    input_layer_type: str,
    feats_dim: int,
    blocks: List[Dict[str, Any]],
    dropout_rate: float,
    pos_enc_dropout_rate: float,
) -> Dict[str, Any]:
    """Prepare input layer arguments.

    Args:
        input_layer_type: Input layer type.
        feats_dim: Dimension of input features.
        blocks: Blocks parameters for network part.
        dropout_rate: Dropout rate for input layer.
        pos_enc_dropout_rate: Dropout rate for input layer pos. enc.

    Return:
        input_block: Input block parameters.

    """
    input_block = {}
    first_block_type = blocks[0].get("type", None)

    if first_block_type == "causal-conv1d":
        input_block["type"] = "c-embed"
    else:
        input_block["type"] = input_layer_type

    input_block["dropout-rate"] = dropout_rate
    input_block["pos-dropout-rate"] = pos_enc_dropout_rate

    input_block["idim"] = feats_dim

    if first_block_type in ("transformer", "conformer"):
        input_block["odim"] = blocks[0].get("d_hidden", 0)
    else:
        input_block["odim"] = blocks[0].get("idim", 0)

    return input_block


def prepare_body_model(
    net_part: str,
    blocks: List[Dict[str, Any]],
) -> Tuple[int]:
    """Prepare model body blocks.

    Args:
        net_part: Network part, either 'encoder' or 'decoder'.
        blocks: Blocks parameters for network part.

    Return:
        : Network output dimension.

    """
    cmp_io = [
        verify_block_arguments(net_part, b, (i + 1)) for i, b in enumerate(blocks)
    ]

    if {"transformer", "conformer"} <= {b["type"] for b in blocks}:
        raise NotImplementedError(
            net_part + ": transformer and conformer blocks "
            "can't be used together in the same net part."
        )

    for i in range(1, len(cmp_io)):
        if cmp_io[(i - 1)][1] != cmp_io[i][0]:
            raise ValueError(
                "Output/Input mismatch between blocks %d and %d in %s"
                % (i, (i + 1), net_part)
            )

    return cmp_io[-1][1]


def get_pos_enc_and_att_class(
    net_part: str, pos_enc_type: str, self_attn_type: str
) -> Tuple[
    Union[PositionalEncoding, ScaledPositionalEncoding, RelPositionalEncoding],
    Union[MultiHeadedAttention, RelPositionMultiHeadedAttention],
]:
    """Get positional encoding and self attention module class.

    Args:
        net_part: Network part, either 'encoder' or 'decoder'.
        pos_enc_type: Positional encoding type.
        self_attn_type: Self-attention type.

    Return:
        pos_enc_class: Positional encoding class.
        self_attn_class: Self-attention class.

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
    block: Dict[str, Any],
    pos_enc_class: torch.nn.Module,
    padding_idx: int,
) -> Tuple[Union[Conv2dSubsampling, VGG2L, torch.nn.Sequential], int]:
    """Build input layer.

    Args:
        block: Architecture definition of input layer.
        pos_enc_class: Positional encoding class.
        padding_idx: Padding symbol ID for embedding layer (if provided).

    Returns:
        : Input layer module.
        subsampling_factor: Subsampling factor.

    """
    input_type = block["type"]

    idim = block["idim"]
    odim = block["odim"]

    dropout_rate = block["dropout-rate"]
    pos_dropout_rate = block["pos-dropout-rate"]

    if pos_enc_class.__name__ == "RelPositionalEncoding":
        pos_enc_class_subsampling = pos_enc_class(odim, pos_dropout_rate)
    else:
        pos_enc_class_subsampling = None

    if input_type == "linear":
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
    elif input_type == "conv2d":
        return Conv2dSubsampling(idim, odim, dropout_rate, pos_enc_class_subsampling), 4
    elif input_type == "vgg2l":
        return VGG2L(idim, odim, pos_enc_class_subsampling), 4
    elif input_type == "embed":
        return (
            torch.nn.Sequential(
                torch.nn.Embedding(idim, odim, padding_idx=padding_idx),
                pos_enc_class(odim, pos_dropout_rate),
            ),
            1,
        )
    elif input_type == "c-embed":
        return (
            torch.nn.Sequential(
                torch.nn.Embedding(idim, odim, padding_idx=padding_idx),
                torch.nn.Dropout(dropout_rate),
            ),
            1,
        )
    else:
        raise NotImplementedError(
            "Invalid input layer: %s. Supported: linear, conv2d, vgg2l and embed"
            % input_type
        )


def build_transformer_block(
    net_part: str,
    block: Dict[str, Any],
    pw_layer_type: str,
    pw_activation_type: str,
) -> Union[EncoderLayer, TransformerDecoderLayer]:
    """Build function for transformer block.

    Args:
        net_part: Network part, either 'encoder' or 'decoder'.
        block: Transformer block parameters.
        pw_layer_type: Positionwise layer type.
        pw_activation_type: Positionwise activation type.

    Returns:
        : Function to create transformer (encoder or decoder) block.

    """
    d_hidden = block["d_hidden"]

    dropout_rate = block.get("dropout-rate", 0.0)
    pos_dropout_rate = block.get("pos-dropout-rate", 0.0)
    att_dropout_rate = block.get("att-dropout-rate", 0.0)

    if pw_layer_type != "linear":
        raise NotImplementedError(
            "Transformer block only supports linear pointwise layer."
        )

    if net_part == "encoder":
        transformer_layer_class = EncoderLayer
    elif net_part == "decoder":
        transformer_layer_class = TransformerDecoderLayer

    return lambda: transformer_layer_class(
        d_hidden,
        MultiHeadedAttention(block["heads"], d_hidden, att_dropout_rate),
        PositionwiseFeedForward(
            d_hidden,
            block["d_ff"],
            pos_dropout_rate,
            get_activation(pw_activation_type),
        ),
        dropout_rate,
    )


def build_conformer_block(
    block: Dict[str, Any],
    self_attn_class: str,
    pw_layer_type: str,
    pw_activation_type: str,
    conv_mod_activation_type: str,
) -> ConformerEncoderLayer:
    """Build function for conformer block.

    Args:
        block: Conformer block parameters.
        self_attn_type: Self-attention module type.
        pw_layer_type: Positionwise layer type.
        pw_activation_type: Positionwise activation type.
        conv_mod_activation_type: Convolutional module activation type.

    Returns:
        : Function to create conformer (encoder) block.

    """
    d_hidden = block["d_hidden"]
    d_ff = block["d_ff"]

    dropout_rate = block.get("dropout-rate", 0.0)
    pos_dropout_rate = block.get("pos-dropout-rate", 0.0)
    att_dropout_rate = block.get("att-dropout-rate", 0.0)

    macaron_style = block["macaron_style"]
    use_conv_mod = block["use_conv_mod"]

    if pw_layer_type == "linear":
        pw_layer = PositionwiseFeedForward
        pw_layer_args = (
            d_hidden,
            d_ff,
            pos_dropout_rate,
            get_activation(pw_activation_type),
        )
    else:
        raise NotImplementedError("Conformer block only supports linear yet.")

    if macaron_style:
        macaron_net = PositionwiseFeedForward
        macaron_net_args = (
            d_hidden,
            d_ff,
            pos_dropout_rate,
            get_activation(pw_activation_type),
        )

    if use_conv_mod:
        conv_mod = ConvolutionModule
        conv_mod_args = (
            d_hidden,
            block["conv_mod_kernel"],
            get_activation(conv_mod_activation_type),
        )

    return lambda: ConformerEncoderLayer(
        d_hidden,
        self_attn_class(block["heads"], d_hidden, att_dropout_rate),
        pw_layer(*pw_layer_args),
        macaron_net(*macaron_net_args) if macaron_style else None,
        conv_mod(*conv_mod_args) if use_conv_mod else None,
        dropout_rate,
    )


def build_conv1d_block(block: Dict[str, Any], block_type: str) -> CausalConv1d:
    """Build function for causal conv1d block.

    Args:
        block: CausalConv1d or Conv1D block parameters.

    Returns:
        : Function to create conv1d (encoder) or causal conv1d (decoder) block.

    """
    if block_type == "conv1d":
        conv_class = Conv1d
    else:
        conv_class = CausalConv1d

    stride = block.get("stride", 1)
    dilation = block.get("dilation", 1)
    groups = block.get("groups", 1)
    bias = block.get("bias", True)

    use_batch_norm = block.get("use-batch-norm", False)
    use_relu = block.get("use-relu", False)
    dropout_rate = block.get("dropout-rate", 0.0)

    return lambda: conv_class(
        block["idim"],
        block["odim"],
        block["kernel_size"],
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=bias,
        relu=use_relu,
        batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
    )


def build_blocks(
    net_part: str,
    idim: int,
    input_layer_type: str,
    blocks: List[Dict[str, Any]],
    repeat_block: int = 0,
    self_attn_type: str = "self_attn",
    positional_encoding_type: str = "abs_pos",
    positionwise_layer_type: str = "linear",
    positionwise_activation_type: str = "relu",
    conv_mod_activation_type: str = "relu",
    input_layer_dropout_rate: float = 0.0,
    input_layer_pos_enc_dropout_rate: float = 0.0,
    padding_idx: int = -1,
) -> Tuple[
    Union[Conv2dSubsampling, VGG2L, torch.nn.Sequential], MultiSequential, int, int
]:
    """Build custom model blocks.

    Args:
        net_part: Network part, either 'encoder' or 'decoder'.
        idim: Input dimension.
        input_layer: Input layer type.
        blocks: Blocks parameters for network part.
        repeat_block: Number of times provided blocks are repeated.
        positional_encoding_type: Positional encoding layer type.
        positionwise_layer_type: Positionwise layer type.
        positionwise_activation_type: Positionwise activation type.
        conv_mod_activation_type: Convolutional module activation type.
        input_layer_dropout_rate: Dropout rate for input layer.
        input_layer_pos_enc_dropout_rate: Dropout rate for input layer pos. enc.
        padding_idx: Padding symbol ID for embedding layer.

    Returns:
        in_layer: Input layer
        all_blocks: Encoder/Decoder network.
        out_dim: Network output dimension.
        conv_subsampling_factor: Subsampling factor in frontend CNN.

    """
    fn_modules = []

    pos_enc_class, self_attn_class = get_pos_enc_and_att_class(
        net_part, positional_encoding_type, self_attn_type
    )

    input_block = prepare_input_layer(
        input_layer_type,
        idim,
        blocks,
        input_layer_dropout_rate,
        input_layer_pos_enc_dropout_rate,
    )

    out_dim = prepare_body_model(net_part, blocks)

    input_layer, conv_subsampling_factor = build_input_layer(
        input_block,
        pos_enc_class,
        padding_idx,
    )

    for i in range(len(blocks)):
        block_type = blocks[i]["type"]

        if block_type in ("causal-conv1d", "conv1d"):
            module = build_conv1d_block(blocks[i], block_type)
        elif block_type == "conformer":
            module = build_conformer_block(
                blocks[i],
                self_attn_class,
                positionwise_layer_type,
                positionwise_activation_type,
                conv_mod_activation_type,
            )
        elif block_type == "transformer":
            module = build_transformer_block(
                net_part,
                blocks[i],
                positionwise_layer_type,
                positionwise_activation_type,
            )

        fn_modules.append(module)

    if repeat_block > 1:
        fn_modules = fn_modules * repeat_block

    return (
        input_layer,
        MultiSequential(*[fn() for fn in fn_modules]),
        out_dim,
        conv_subsampling_factor,
    )
