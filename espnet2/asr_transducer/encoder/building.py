"""Set of methods to build Transducer encoder architecture."""

from typing import Any, Dict, List, Union

from torch.nn import LayerNorm

from espnet2.asr_transducer.activation import get_activation
from espnet2.asr_transducer.encoder.blocks.conformer import Conformer
from espnet2.asr_transducer.encoder.blocks.conv1d import Conv1d
from espnet2.asr_transducer.encoder.blocks.conv_input import ConvInput
from espnet2.asr_transducer.encoder.modules.attention import (  # noqa: H301
    RelPositionMultiHeadedAttention,
)
from espnet2.asr_transducer.encoder.modules.convolution import (  # noqa: H301
    ConformerConvolution,
)
from espnet2.asr_transducer.encoder.modules.multi_blocks import MultiBlocks
from espnet2.asr_transducer.encoder.modules.positional_encoding import (  # noqa: H301
    RelPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


def build_main_parameters(
    pos_wise_act_type: str = "swish",
    conv_mod_act_type: str = "swish",
    pos_enc_dropout_rate: float = 0.0,
    pos_enc_max_len: int = 5000,
    dynamic_chunk_training: bool = False,
    short_chunk_threshold: float = 0.75,
    short_chunk_size: int = 25,
    left_chunk_size: int = 0,
    **activation_parameters,
) -> Dict[str, Any]:
    """Build encoder main parameters.

    Args:
        pos_wise_act_type: Position-wise activation type.
        conv_mod_act_type: Convolutional module activation type.
        pos_enc_dropout_rate: Positional encoding dropout rate.
        pos_enc_max_len: Positional encoding maximum length.
        dynamic_chunk_training: Whether to use dynamic chunk training.
        short_chunk_threshold: Threshold for dynamic chunk selection.
        short_chunk_size: Minimum number of frames during dynamic chunk training.
        left_chunk_size: Number of frames in left context.
        **activations_parameters: Parameters of the activation functions.
                                    (See espnet2/asr_transducer/activation.py)

    Returns:
        : Main encoder parameters

    """
    main_params = {}

    main_params["pos_wise_act"] = get_activation(
        pos_wise_act_type, **activation_parameters
    )
    main_params["conv_mod_act"] = get_activation(
        conv_mod_act_type, **activation_parameters
    )

    main_params["pos_enc_dropout_rate"] = pos_enc_dropout_rate
    main_params["pos_enc_max_len"] = pos_enc_max_len

    main_params["dynamic_chunk_training"] = dynamic_chunk_training
    main_params["short_chunk_threshold"] = max(0, short_chunk_threshold)
    main_params["short_chunk_size"] = max(0, short_chunk_size)
    main_params["left_chunk_size"] = max(0, left_chunk_size)

    return main_params


def build_positional_encoding(
    block_size: int, configuration: Dict[str, Any]
) -> RelPositionalEncoding:
    """Build positional encoding block.

    Args:
        block_size: Input/output size.
        configuration: Positional encoding configuration.

    Returns:
        : Positional encoding module.

    """
    return RelPositionalEncoding(
        block_size,
        configuration.get("pos_enc_dropout_rate", 0.0),
        max_len=configuration.get("pos_enc_max_len", 5000),
    )


def build_input_block(
    input_size: int,
    configuration: Dict[str, Union[str, int]],
) -> ConvInput:
    """Build encoder input block.

    Args:
        input_size: Input size.
        configuration: Input block configuration.

    Returns:
        : ConvInput block function.

    """
    return ConvInput(
        input_size,
        configuration["conv_size"],
        configuration["subsampling_factor"],
        vgg_like=configuration["vgg_like"],
        output_size=configuration["output_size"],
    )


def build_conformer_block(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
) -> Conformer:
    """Build conformer block.

    Args:
        configuration: Conformer block configuration.
        main_params: Encoder main parameters.

    Returns:
        : Conformer block function.

    """
    hidden_size = configuration["hidden_size"]
    linear_size = configuration["linear_size"]

    pos_wise_args = (
        hidden_size,
        linear_size,
        configuration.get("pos_wise_dropout_rate", 0.0),
        main_params["pos_wise_act"],
    )

    conv_args = (
        hidden_size,
        configuration["conv_mod_kernel_size"],
        main_params["conv_mod_act"],
        main_params["dynamic_chunk_training"],
    )

    mult_att_args = (
        configuration.get("heads", 4),
        hidden_size,
        configuration.get("att_dropout_rate", 0.0),
    )

    layer_norm_class = LayerNorm
    layer_norm_eps = configuration.get("layer_norm_eps", 1e-12)

    return lambda: Conformer(
        hidden_size,
        RelPositionMultiHeadedAttention(*mult_att_args),
        PositionwiseFeedForward(*pos_wise_args),
        PositionwiseFeedForward(*pos_wise_args),
        ConformerConvolution(*conv_args),
        layer_norm=layer_norm_class,
        layer_norm_eps=layer_norm_eps,
        dropout_rate=configuration.get("dropout_rate", 0.0),
    )


def build_conv1d_block(
    configuration: List[Dict[str, Any]],
    causal: bool,
) -> Conv1d:
    """Build Conv1d block.

    Args:
        configuration: Conv1d block configuration.

    Returns:
        : Conv1d block function.

    """
    return lambda: Conv1d(
        configuration["input_size"],
        configuration["output_size"],
        configuration["kernel_size"],
        stride=configuration.get("stride", 1),
        dilation=configuration.get("dilation", 1),
        groups=configuration.get("groups", 1),
        bias=configuration.get("bias", True),
        relu=configuration.get("use_relu", True),
        batch_norm=configuration.get("use_batchnorm", False),
        causal=causal,
        dropout_rate=configuration.get("dropout_rate", 0.0),
    )


def build_body_blocks(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
    output_size: int,
) -> MultiBlocks:
    """Build encoder body blocks.

    Args:
        configuration: Body blocks configuration.
        main_params: Encoder main parameters.
        output_size: Architecture output size.

    Returns:
        MultiBlocks function encapsulation all encoder blocks.

    """
    fn_modules = []
    extended_conf = []

    for c in configuration:
        if c.get("num_blocks") is not None:
            extended_conf += c["num_blocks"] * [
                {c_i: c[c_i] for c_i in c if c_i != "num_blocks"}
            ]
        else:
            extended_conf += [c]

    for i, c in enumerate(extended_conf):
        block_type = c["block_type"]

        if block_type == "conv1d":
            module = build_conv1d_block(c, main_params["dynamic_chunk_training"])
        elif block_type == "conformer":
            module = build_conformer_block(c, main_params)
        else:
            raise NotImplementedError

        fn_modules.append(module)

    return MultiBlocks([fn() for fn in fn_modules], output_size)
