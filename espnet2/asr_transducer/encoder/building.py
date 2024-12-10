"""Set of methods to build Transducer encoder architecture."""

from typing import Any, Dict, List, Optional, Union

from espnet2.asr_transducer.activation import get_activation
from espnet2.asr_transducer.encoder.blocks.branchformer import Branchformer
from espnet2.asr_transducer.encoder.blocks.conformer import Conformer
from espnet2.asr_transducer.encoder.blocks.conv1d import Conv1d
from espnet2.asr_transducer.encoder.blocks.conv_input import ConvInput
from espnet2.asr_transducer.encoder.blocks.ebranchformer import EBranchformer
from espnet2.asr_transducer.encoder.modules.attention import (  # noqa: H301
    RelPositionMultiHeadedAttention,
)
from espnet2.asr_transducer.encoder.modules.convolution import (  # noqa: H301
    ConformerConvolution,
    ConvolutionalSpatialGatingUnit,
    DepthwiseConvolution,
)
from espnet2.asr_transducer.encoder.modules.multi_blocks import MultiBlocks
from espnet2.asr_transducer.encoder.modules.positional_encoding import (  # noqa: H301
    RelPositionalEncoding,
)
from espnet2.asr_transducer.normalization import get_normalization
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


def build_main_parameters(
    pos_wise_act_type: str = "swish",
    conv_mod_act_type: str = "swish",
    pos_enc_dropout_rate: float = 0.0,
    pos_enc_max_len: int = 5000,
    simplified_att_score: bool = False,
    norm_type: str = "layer_norm",
    conv_mod_norm_type: str = "layer_norm",
    after_norm_eps: Optional[float] = None,
    after_norm_partial: Optional[float] = None,
    blockdrop_rate: float = 0.0,
    dynamic_chunk_training: bool = False,
    short_chunk_threshold: float = 0.75,
    short_chunk_size: int = 25,
    num_left_chunks: int = 0,
    **activation_parameters,
) -> Dict[str, Any]:
    """
    Build encoder main parameters.

    This function constructs the main parameters required for the encoder
    architecture of a Transducer model, allowing customization of various
    components such as activation functions, normalization types, and 
    dropout rates.

    Args:
        pos_wise_act_type: X-former position-wise feed-forward activation type.
        conv_mod_act_type: X-former convolution module activation type.
        pos_enc_dropout_rate: Positional encoding dropout rate.
        pos_enc_max_len: Positional encoding maximum length.
        simplified_att_score: Whether to use simplified attention score 
                              computation.
        norm_type: X-former normalization module type.
        conv_mod_norm_type: Conformer convolution module normalization type.
        after_norm_eps: Epsilon value for the final normalization.
        after_norm_partial: Value for the final normalization with RMSNorm.
        blockdrop_rate: Probability threshold of dropping out each encoder block.
        dynamic_chunk_training: Whether to use dynamic chunk training.
        short_chunk_threshold: Threshold for dynamic chunk selection.
        short_chunk_size: Minimum number of frames during dynamic chunk 
                          training.
        num_left_chunks: Number of left chunks the attention module can see. 
                         (null or negative value means full context)
        **activation_parameters: Parameters of the activation functions.
                                 (See espnet2/asr_transducer/activation.py)

    Returns:
        dict: Main encoder parameters including activation functions, 
              dropout rates, normalization configurations, and other 
              settings required for the encoder.

    Examples:
        >>> params = build_main_parameters(
        ...     pos_wise_act_type='relu',
        ...     blockdrop_rate=0.1,
        ...     dynamic_chunk_training=True
        ... )
        >>> print(params)
        {
            'pos_wise_act': <activation_function>,
            'conv_mod_act': <activation_function>,
            'pos_enc_dropout_rate': 0.0,
            ...
        }
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

    main_params["simplified_att_score"] = simplified_att_score

    main_params["norm_type"] = norm_type
    main_params["conv_mod_norm_type"] = conv_mod_norm_type

    (
        main_params["after_norm_class"],
        main_params["after_norm_args"],
    ) = get_normalization(norm_type, eps=after_norm_eps, partial=after_norm_partial)

    main_params["blockdrop_rate"] = blockdrop_rate

    main_params["dynamic_chunk_training"] = dynamic_chunk_training
    main_params["short_chunk_threshold"] = max(0, short_chunk_threshold)
    main_params["short_chunk_size"] = max(0, short_chunk_size)
    main_params["num_left_chunks"] = max(0, num_left_chunks)

    return main_params


def build_positional_encoding(
    block_size: int, configuration: Dict[str, Any]
) -> RelPositionalEncoding:
    """
    Build positional encoding block.

    This function creates a positional encoding module, which is used in 
    transformer architectures to inject information about the position of 
    tokens in the input sequence. The positional encoding helps the model 
    understand the order of tokens.

    Args:
        block_size: Input/output size of the positional encoding.
        configuration: A dictionary containing the positional encoding 
                       configuration parameters.

    Returns:
        RelPositionalEncoding: An instance of the positional encoding module.

    Examples:
        >>> config = {'pos_enc_dropout_rate': 0.1, 'pos_enc_max_len': 1000}
        >>> pos_enc = build_positional_encoding(512, config)
        >>> print(pos_enc)
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
    """
    Build encoder input block.

    This function constructs the input block for the encoder, which typically
    includes a convolutional layer for processing input features. The configuration
    dictates the specifics of the convolutional layer, such as size and
    subsampling factors.

    Args:
        input_size: The size of the input features.
        configuration: A dictionary containing the input block configuration,
            which must include the following keys:
            - 'conv_size': Size of the convolutional layer.
            - 'subsampling_factor': Factor by which to subsample the input.
            - 'vgg_like': Boolean indicating whether to use VGG-like architecture.
            - 'output_size': Size of the output features after the input block.

    Returns:
        ConvInput: An instance of the ConvInput block configured as specified.

    Examples:
        >>> config = {
        ...     'conv_size': 3,
        ...     'subsampling_factor': 2,
        ...     'vgg_like': True,
        ...     'output_size': 128
        ... }
        >>> input_block = build_input_block(256, config)
        >>> print(input_block)
    """
    return ConvInput(
        input_size,
        configuration["conv_size"],
        configuration["subsampling_factor"],
        vgg_like=configuration["vgg_like"],
        output_size=configuration["output_size"],
    )


def build_branchformer_block(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
) -> Branchformer:
    """
    Build Branchformer block.

    This function constructs a Branchformer block, which is a component of
    the encoder architecture in the Transducer model. The Branchformer 
    block leverages attention mechanisms and convolutional layers to 
    process input data efficiently.

    Args:
        configuration: A list of dictionaries containing the configuration 
                       for the Branchformer block. Each dictionary must 
                       include the keys:
                       - hidden_size: Size of the hidden layer.
                       - linear_size: Size of the linear layer.
                       - conv_mod_kernel_size: Kernel size for the convolutional 
                         module.
                       - dropout_rate: Dropout rate for the block.
                       - heads: Number of attention heads (optional, default is 4).
                       - att_dropout_rate: Dropout rate for attention (optional).
                       - norm_eps: Epsilon value for normalization (optional).
                       - norm_partial: Partial value for normalization (optional).
        main_params: A dictionary containing the main parameters for the 
                     encoder, including:
                     - conv_mod_norm_type: Type of normalization for the 
                       convolution module.
                     - simplified_att_score: Boolean indicating if simplified 
                       attention scoring is used.

    Returns:
        Branchformer: A callable function that returns a Branchformer block 
                      when invoked.

    Examples:
        >>> config = [
        ...     {
        ...         "hidden_size": 256,
        ...         "linear_size": 128,
        ...         "conv_mod_kernel_size": 3,
        ...         "dropout_rate": 0.1,
        ...         "heads": 4
        ...     }
        ... ]
        >>> main_params = {
        ...     "conv_mod_norm_type": "layer_norm",
        ...     "simplified_att_score": False
        ... }
        >>> branchformer_block = build_branchformer_block(config, main_params)
        >>> block = branchformer_block()  # Instantiate the block
    """
    hidden_size = configuration["hidden_size"]
    linear_size = configuration["linear_size"]

    dropout_rate = configuration.get("dropout_rate", 0.0)

    conv_mod_norm_class, conv_mod_norm_args = get_normalization(
        main_params["conv_mod_norm_type"],
        eps=configuration.get("conv_mod_norm_eps"),
        partial=configuration.get("conv_mod_norm_partial"),
    )

    conv_mod_args = (
        linear_size,
        configuration["conv_mod_kernel_size"],
        conv_mod_norm_class,
        conv_mod_norm_args,
        dropout_rate,
        main_params["dynamic_chunk_training"],
    )

    mult_att_args = (
        configuration.get("heads", 4),
        hidden_size,
        configuration.get("att_dropout_rate", 0.0),
        main_params["simplified_att_score"],
    )

    norm_class, norm_args = get_normalization(
        main_params["norm_type"],
        eps=configuration.get("norm_eps"),
        partial=configuration.get("norm_partial"),
    )

    return lambda: Branchformer(
        hidden_size,
        linear_size,
        RelPositionMultiHeadedAttention(*mult_att_args),
        ConvolutionalSpatialGatingUnit(*conv_mod_args),
        norm_class=norm_class,
        norm_args=norm_args,
        dropout_rate=dropout_rate,
    )


def build_conformer_block(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
) -> Conformer:
    """
    Build Conformer block.

    This function constructs a Conformer block based on the provided 
    configuration and main parameters. The Conformer architecture 
    integrates convolutional layers and attention mechanisms to 
    capture both local and global dependencies in the input data.

    Args:
        configuration: A list of dictionaries containing Conformer 
                       block configuration settings. Each dictionary 
                       should include keys such as "hidden_size", 
                       "linear_size", "pos_wise_dropout_rate", and 
                       "conv_mod_kernel_size".
        main_params: A dictionary containing encoder main parameters, 
                     which are used to configure the various components 
                     of the Conformer block, including activation functions 
                     and normalization settings.

    Returns:
        A callable that returns a Conformer block instance when invoked.

    Examples:
        >>> config = [{
        ...     "hidden_size": 256,
        ...     "linear_size": 512,
        ...     "pos_wise_dropout_rate": 0.1,
        ...     "conv_mod_kernel_size": 31,
        ...     "heads": 4,
        ...     "att_dropout_rate": 0.1,
        ...     "dropout_rate": 0.1,
        ...     "norm_eps": 1e-5
        ... }]
        >>> main_params = {
        ...     "pos_wise_act": "relu",
        ...     "conv_mod_act": "relu",
        ...     "norm_type": "layer_norm",
        ...     "conv_mod_norm_type": "layer_norm",
        ...     "dynamic_chunk_training": False
        ... }
        >>> conformer_block = build_conformer_block(config, main_params)()
    """
    hidden_size = configuration["hidden_size"]
    linear_size = configuration["linear_size"]

    pos_wise_args = (
        hidden_size,
        linear_size,
        configuration.get("pos_wise_dropout_rate", 0.0),
        main_params["pos_wise_act"],
    )

    conv_mod_norm_args = {
        "eps": configuration.get("conv_mod_norm_eps", 1e-05),
        "momentum": configuration.get("conv_mod_norm_momentum", 0.1),
    }

    conv_mod_args = (
        hidden_size,
        configuration["conv_mod_kernel_size"],
        main_params["conv_mod_act"],
        conv_mod_norm_args,
        main_params["dynamic_chunk_training"],
    )

    mult_att_args = (
        configuration.get("heads", 4),
        hidden_size,
        configuration.get("att_dropout_rate", 0.0),
        main_params["simplified_att_score"],
    )

    norm_class, norm_args = get_normalization(
        main_params["norm_type"],
        eps=configuration.get("norm_eps"),
        partial=configuration.get("norm_partial"),
    )

    return lambda: Conformer(
        hidden_size,
        RelPositionMultiHeadedAttention(*mult_att_args),
        PositionwiseFeedForward(*pos_wise_args),
        PositionwiseFeedForward(*pos_wise_args),
        ConformerConvolution(*conv_mod_args),
        norm_class=norm_class,
        norm_args=norm_args,
        dropout_rate=configuration.get("dropout_rate", 0.0),
    )


def build_conv1d_block(
    configuration: List[Dict[str, Any]],
    causal: bool,
) -> Conv1d:
    """
    Build Conv1d block.

    This function constructs a Conv1d block based on the provided configuration
    and causal flag. The Conv1d block is used in various neural network architectures
    for processing sequential data.

    Args:
        configuration: A list of dictionaries where each dictionary contains 
                       the configuration parameters for the Conv1d block. 
                       Expected keys include:
                       - input_size: Size of the input features.
                       - output_size: Size of the output features.
                       - kernel_size: Size of the convolutional kernel.
                       - stride: Stride of the convolution (default is 1).
                       - dilation: Dilation factor (default is 1).
                       - groups: Number of groups for grouped convolution 
                                 (default is 1).
                       - bias: Whether to include a bias term (default is True).
                       - relu: Whether to apply ReLU activation (default is True).
                       - batch_norm: Whether to include batch normalization 
                                     (default is False).
        causal: A boolean indicating whether the convolution should be causal. 
                If True, the convolution will not include future time steps.

    Returns:
        A Conv1d block function that can be called to create the Conv1d layer.

    Examples:
        >>> conv1d_block = build_conv1d_block(
        ...     configuration=[
        ...         {
        ...             "input_size": 128,
        ...             "output_size": 256,
        ...             "kernel_size": 3,
        ...             "stride": 1,
        ...             "dilation": 1,
        ...             "groups": 1,
        ...             "bias": True,
        ...             "relu": True,
        ...             "batch_norm": False,
        ...         }
        ...     ],
        ...     causal=True
        ... )
        >>> conv_layer = conv1d_block()
    """
    return lambda: Conv1d(
        configuration["input_size"],
        configuration["output_size"],
        configuration["kernel_size"],
        stride=configuration.get("stride", 1),
        dilation=configuration.get("dilation", 1),
        groups=configuration.get("groups", 1),
        bias=configuration.get("bias", True),
        relu=configuration.get("relu", True),
        batch_norm=configuration.get("batch_norm", False),
        causal=causal,
        dropout_rate=configuration.get("dropout_rate", 0.0),
    )


def build_ebranchformer_block(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
) -> EBranchformer:
    """
    Build E-Branchformer block.

    This function constructs an E-Branchformer block, which is part of the 
    encoder architecture for transducers. It utilizes various configurations 
    and main parameters to create a functional block for encoding input data.

    Args:
        configuration: A list of dictionaries containing the E-Branchformer 
            block configuration parameters such as `hidden_size`, `linear_size`, 
            `dropout_rate`, `pos_wise_dropout_rate`, `conv_mod_kernel_size`, 
            `heads`, `att_dropout_rate`, `depth_conv_kernel_size`, `norm_eps`, 
            and `norm_partial`.
        main_params: A dictionary containing the main encoder parameters, 
            including activation functions, normalization types, and dropout 
            rates.

    Returns:
        EBranchformer: A callable that constructs an E-Branchformer block 
            function when invoked.

    Examples:
        config = [{
            "hidden_size": 256,
            "linear_size": 512,
            "dropout_rate": 0.1,
            "pos_wise_dropout_rate": 0.1,
            "conv_mod_kernel_size": 3,
            "heads": 4,
            "att_dropout_rate": 0.1,
            "depth_conv_kernel_size": 3,
            "norm_eps": 1e-5,
            "norm_partial": None
        }]
        main_params = build_main_parameters()
        e_branchformer_block = build_ebranchformer_block(config, main_params)()
    """
    hidden_size = configuration["hidden_size"]
    linear_size = configuration["linear_size"]

    dropout_rate = configuration.get("dropout_rate", 0.0)

    pos_wise_args = (
        hidden_size,
        linear_size,
        configuration.get("pos_wise_dropout_rate", 0.0),
        main_params["pos_wise_act"],
    )

    conv_mod_norm_class, conv_mod_norm_args = get_normalization(
        main_params["conv_mod_norm_type"],
        eps=configuration.get("conv_mod_norm_eps"),
        partial=configuration.get("conv_mod_norm_partial"),
    )

    conv_mod_args = (
        linear_size,
        configuration["conv_mod_kernel_size"],
        conv_mod_norm_class,
        conv_mod_norm_args,
        dropout_rate,
        main_params["dynamic_chunk_training"],
    )

    mult_att_args = (
        configuration.get("heads", 4),
        hidden_size,
        configuration.get("att_dropout_rate", 0.0),
        main_params["simplified_att_score"],
    )

    depthwise_conv_args = (
        hidden_size,
        configuration.get(
            "depth_conv_kernel_size", configuration["conv_mod_kernel_size"]
        ),
        main_params["dynamic_chunk_training"],
    )

    norm_class, norm_args = get_normalization(
        main_params["norm_type"],
        eps=configuration.get("norm_eps"),
        partial=configuration.get("norm_partial"),
    )

    return lambda: EBranchformer(
        hidden_size,
        linear_size,
        RelPositionMultiHeadedAttention(*mult_att_args),
        PositionwiseFeedForward(*pos_wise_args),
        PositionwiseFeedForward(*pos_wise_args),
        ConvolutionalSpatialGatingUnit(*conv_mod_args),
        DepthwiseConvolution(*depthwise_conv_args),
        norm_class=norm_class,
        norm_args=norm_args,
        dropout_rate=dropout_rate,
    )


def build_body_blocks(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
    output_size: int,
) -> MultiBlocks:
    """
    Build encoder body blocks.

    This function constructs a series of encoder body blocks based on the given
    configuration and main parameters. It allows for the creation of different
    types of blocks, including Branchformer, Conformer, Conv1d, and E-Branchformer,
    according to the specified configuration.

    Args:
        configuration: A list of dictionaries containing the configuration for each
                       body block. Each dictionary may specify the type of block,
                       the number of blocks, and various hyperparameters.
        main_params: A dictionary containing the main parameters for the encoder,
                     including activation functions, normalization types, and other
                     relevant settings.
        output_size: The output size of the architecture after processing through
                     the body blocks.

    Returns:
        MultiBlocks: A function encapsulating all encoder blocks, which can be
                     invoked to create the full encoder body.

    Raises:
        NotImplementedError: If an unsupported block type is specified in the
                             configuration.

    Examples:
        configuration = [
            {"block_type": "branchformer", "num_blocks": 2, "hidden_size": 256,
             "linear_size": 128, "dropout_rate": 0.1},
            {"block_type": "conformer", "num_blocks": 1, "hidden_size": 256,
             "linear_size": 128, "pos_wise_dropout_rate": 0.1},
        ]
        main_params = {
            "pos_wise_act": "relu",
            "norm_type": "layer_norm",
            "after_norm_class": "layer_norm",
            "after_norm_args": {"eps": 1e-6},
            "blockdrop_rate": 0.0,
        }
        output_size = 512

        body_blocks = build_body_blocks(configuration, main_params, output_size)
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

        if block_type == "branchformer":
            module = build_branchformer_block(c, main_params)
        elif block_type == "conformer":
            module = build_conformer_block(c, main_params)
        elif block_type == "conv1d":
            module = build_conv1d_block(c, main_params["dynamic_chunk_training"])
        elif block_type == "ebranchformer":
            module = build_ebranchformer_block(c, main_params)
        else:
            raise NotImplementedError

        fn_modules.append(module)

    return MultiBlocks(
        [fn() for fn in fn_modules],
        output_size,
        norm_class=main_params["after_norm_class"],
        norm_args=main_params["after_norm_args"],
        blockdrop_rate=main_params["blockdrop_rate"],
    )
