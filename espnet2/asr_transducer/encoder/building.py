"""Set of methods to build Transducer encoder architecture."""

from typing import Any
from typing import Dict
from typing import List
from typing import Union

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
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
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from espnet2.asr_transducer.activation import get_activation
from espnet2.asr_transducer.encoder.blocks.conformer import Conformer
from espnet2.asr_transducer.encoder.blocks.conv1d import Conv1d
from espnet2.asr_transducer.encoder.blocks.conv2d_subsampling import (
    Conv2dSubsampling,  # noqa: H301
)
from espnet2.asr_transducer.encoder.blocks.rnn import RNN
from espnet2.asr_transducer.encoder.blocks.rnnp import RNNP
from espnet2.asr_transducer.encoder.blocks.vgg2l import VGG2L
from espnet2.train.class_choices import ClassChoices

pos_enc_choices = ClassChoices(
    "positional_encoding",
    classes=dict(
        abs_pos=PositionalEncoding,
        scaled_abs_pos=ScaledPositionalEncoding,
        rel_pos=RelPositionalEncoding,
    ),
    default="abs_pos",
)
pos_wise_choices = ClassChoices(
    "position_wise",
    classes=dict(
        linear=PositionwiseFeedForward,
        conv1d=MultiLayeredConv1d,
        conv1d_linear=Conv1dLinear,
    ),
    default="linear",
)
selfattn_choices = ClassChoices(
    "self_attention",
    classes=dict(
        selfattn=MultiHeadedAttention,
        rel_selfattn=RelPositionMultiHeadedAttention,
    ),
    default="selfattn",
)
input_block_choices = ClassChoices(
    "input_block",
    classes=dict(
        conv2d=Conv2dSubsampling,
        vgg=VGG2L,
    ),
    default="conv2d",
)


def build_main_parameters(
    need_pos: bool,
    pos_enc_layer_type: str = "abs_pos",
    pos_wise_layer_type: str = "linear",
    pos_wise_act_type: str = "swish",
    conv_mod_act_type: str = "swish",
    **activation_parameters,
) -> Dict[str, Any]:
    """Build encoder main parameters.

    Args:
        need_pos: Whether position-related information are needed.
        pos_enc_layer_type: Positional encoding layer type.
        pos_wise_layer_type: Position-wise layer type.
        pos_wise_act_type: Position-wise activation type.
        conv_mod_act_type: Convolutional module activation type.

    Returns:
        : Main encoder parameters
        : Type of mask for forward computation.

    """
    if not need_pos:
        return {"pos_enc_class": None}, "rnn"

    main_params = {}

    main_params["pos_enc_class"] = pos_enc_choices.get_class(pos_enc_layer_type)
    main_params["pos_wise_class"] = pos_wise_choices.get_class(pos_wise_layer_type)
    main_params["selfattn_class"] = selfattn_choices.get_class(
        "rel_selfattn" if "rel" in pos_enc_layer_type else "selfattn"
    )

    main_params["pos_wise_act"] = get_activation(
        pos_wise_act_type, **activation_parameters
    )
    main_params["conv_mod_act"] = get_activation(
        conv_mod_act_type, **activation_parameters
    )

    return main_params, "conformer"


def build_input_block(
    dim_input: int,
    configuration: Dict[str, Union[str, int]],
    mask_type: str,
    pos_enc_class: Union[
        PositionalEncoding, ScaledPositionalEncoding, RelPositionalEncoding
    ] = None,
) -> Union[Conv2dSubsampling, VGG2L]:
    """Build encoder input block.

    Args:
        dim_input: Input dimension.
        dim_output: Output dimension.
        configuration: Input block configuration.
        mask_type: Type of mask for forward computation.
        pos_enc_class: Positional encoding class.

    Returns:
        : Input block.

    """
    block_type = configuration["block_type"]
    dim_output = configuration["dim_output"]

    if pos_enc_class is not None:
        pos_enc = pos_enc_class(
            dim_output if dim_output is not None else configuration["dim_pos_enc"],
            configuration.get("dropout_rate_pos_enc", 0.0),
        )
    else:
        pos_enc = None

    input_class = input_block_choices.get_class(block_type)

    if block_type == "conv2d":
        _conv_param = {"dim_conv": configuration["dim_conv"]}
    else:
        _conv_param = {}

    return input_class(
        dim_input,
        mask_type,
        **_conv_param,
        subsampling_factor=configuration.get("subsampling_factor", 4),
        pos_enc=pos_enc,
        dim_output=dim_output,
    )


def build_conformer_block(
    configuration: List[Dict[str, Any]], main_params: Dict[str, Any]
) -> Conformer:
    """Build conformer block.

    Args:
        configuration: Conformer block configuration.
        main_params: Encoder main parameters.

    Returns:
        : Conformer block function.

    """
    dim_hidden = configuration["dim_hidden"]
    dim_linear = configuration["dim_linear"]

    dropout_rate_pos_wise = configuration.get("dropout_rate_pos_wise", 0.0)
    pos_wise_act = main_params["pos_wise_act"]

    pos_wise_layer = main_params["pos_wise_class"]
    pos_wise_args = (
        dim_hidden,
        dim_linear,
        dropout_rate_pos_wise,
        pos_wise_act,
    )

    conv_mod_kernel = configuration.get("conv_mod_kernel", 0)
    use_conv_mod = conv_mod_kernel > 0
    use_macaron = configuration.get("macaron_style", False)

    if use_macaron:
        macaron_net = main_params["pos_wise_class"]
        macaron_args = (
            dim_hidden,
            dim_linear,
            dropout_rate_pos_wise,
            pos_wise_act,
        )

    if use_conv_mod:
        conv_mod = ConvolutionModule
        conv_args = (dim_hidden, conv_mod_kernel, main_params["conv_mod_act"])

    return lambda: Conformer(
        dim_hidden,
        main_params["selfattn_class"](
            configuration.get("heads", 4),
            dim_hidden,
            configuration.get("dropout_rate_att", 0.0),
        ),
        pos_wise_layer(*pos_wise_args),
        macaron_net(*macaron_args) if use_macaron else None,
        conv_mod(*conv_args) if use_conv_mod else None,
        dropout_rate=configuration.get("dropout_rate", 0.0),
        eps_layer_norm=configuration.get("eps_layer_norm", 1e-12),
    )


def build_conv1d_block(configuration: List[Dict[str, Any]], mask_type: str) -> Conv1d:
    """Build Conv1d block.

    Args:
        configuration: Conv1d block configuration.
        mask_type: Type of mask for forward computation.

    Returns:
        : Conv1d block function.

    """
    return lambda: Conv1d(
        configuration["dim_input"],
        configuration["dim_output"],
        configuration["kernel_size"],
        mask_type,
        stride=configuration.get("stride", 1),
        dilation=configuration.get("dilation", 1),
        groups=configuration.get("groups", 1),
        bias=configuration.get("bias", True),
        relu=configuration.get("use_relu", True),
        batch_norm=configuration.get("use_batchnorm", False),
        dropout_rate=configuration.get("dropout_rate", 0.0),
    )


def build_rnn_block(configuration: List[Dict[str, Any]]) -> Union[RNN, RNNP]:
    """Build RNN block.

    Args:
        configuration: RNN block configuration.

    Returns:
        : RNN/RNNP block function.

    """
    dim_proj = configuration.get("dim_proj")

    if dim_proj is not None:
        return lambda: RNNP(
            configuration["dim_input"],
            configuration["dim_hidden"],
            configuration["dim_proj"],
            rnn_type=configuration.get("rnn_type", "lstm"),
            bidirectional=configuration.get("bidirectional", True),
            num_blocks=configuration.get("num_blocks", 1),
            dropout_rate=configuration.get("dropout_rate", 0.0),
            subsample=configuration.get("subsample"),
            dim_output=configuration.get("dim_output"),
        )

    return lambda: RNN(
        configuration["dim_input"],
        configuration["dim_hidden"],
        rnn_type=configuration.get("rnn_type", "lstm"),
        bidirectional=configuration.get("bidirectional", True),
        num_blocks=configuration.get("num_blocks", 1),
        dim_output=configuration.get("dim_output"),
        dropout_rate=configuration.get("dropout_rate", 0.0),
    )


def build_body_blocks(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
    mask_type: str,
) -> MultiSequential:
    """Build encoder body blocks.

    Args:
        configuration: Body blocks configuration.
        main_params: Encoder main parameters.
        mask_type: Type of mask for forward computation.

    Returns:
        Encoder container.

    """
    fn_modules = []
    extended_conf = []

    for c in configuration:
        if c["block_type"] != "rnn" and c.get("num_blocks") is not None:
            extended_conf += c["num_blocks"] * [
                {c_i: c[c_i] for c_i in c if c_i != "num_blocks"}
            ]
        else:
            extended_conf += [c]

    for c in extended_conf:
        block_type = c["block_type"]

        if block_type == "conv1d":
            module = build_conv1d_block(c, mask_type)
        elif block_type == "conformer":
            module = build_conformer_block(c, main_params)
        elif block_type == "rnn":
            module = build_rnn_block(c)

        fn_modules.append(module)

    return MultiSequential(*[fn() for fn in fn_modules])
