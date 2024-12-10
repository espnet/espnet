"""Set of methods to validate encoder architecture."""

from typing import Any, Dict, List, Tuple

from espnet2.asr_transducer.utils import get_convinput_module_parameters


def validate_block_arguments(
    configuration: Dict[str, Any],
    block_id: int,
    previous_block_output: int,
) -> Tuple[int, int]:
    """
    Set of methods to validate encoder architecture.

    This module provides a set of functions to validate the configuration 
    of various blocks in an encoder architecture, ensuring that all required 
    parameters are present and correctly defined.

    Functions:
        validate_block_arguments(configuration, block_id, previous_block_output):
            Validate block arguments for a given block configuration.
        
        validate_input_block(configuration, body_first_conf, input_size):
            Validate the input block configuration and its output size.
        
        validate_architecture(input_conf, body_conf, input_size):
            Validate the entire architecture by checking input and body block 
            configurations.
    """
    block_type = configuration.get("block_type")

    if block_type is None:
        raise ValueError(
            "Block %d in encoder doesn't have a type assigned. " % block_id
        )

    if block_type in ["branchformer", "conformer", "ebranchformer"]:
        if configuration.get("linear_size") is None:
            raise ValueError(
                "Missing 'linear_size' argument for X-former block (ID: %d)" % block_id
            )

        if configuration.get("conv_mod_kernel_size") is None:
            raise ValueError(
                "Missing 'conv_mod_kernel_size' argument for X-former block (ID: %d)"
                % block_id
            )

        input_size = configuration.get("hidden_size")
        output_size = configuration.get("hidden_size")

    elif block_type == "conv1d":
        output_size = configuration.get("output_size")

        if output_size is None:
            raise ValueError(
                "Missing 'output_size' argument for Conv1d block (ID: %d)" % block_id
            )

        if configuration.get("kernel_size") is None:
            raise ValueError(
                "Missing 'kernel_size' argument for Conv1d block (ID: %d)" % block_id
            )

        input_size = configuration["input_size"] = previous_block_output
    else:
        raise ValueError("Block type: %s is not supported." % block_type)

    return input_size, output_size


def validate_input_block(
    configuration: Dict[str, Any], body_first_conf: Dict[str, Any], input_size: int
) -> int:
    """
    Validate the input block configuration for the encoder architecture.

    This function checks the validity of the input block configuration against the
    first body block configuration. It ensures that the parameters are correctly set
    and compatible with the specified block types and subsampling factors.

    Attributes:
        configuration (Dict[str, Any]): Encoder input block configuration.
        body_first_conf (Dict[str, Any]): Encoder first body block configuration.
        input_size (int): Encoder input block input size.

    Args:
        configuration: A dictionary containing the configuration for the encoder
            input block.
        body_first_conf: A dictionary containing the configuration for the first
            body block of the encoder.
        input_size: An integer representing the input size for the encoder input
            block.

    Returns:
        int: The output size of the encoder input block.

    Raises:
        ValueError: If the subsampling factor is invalid for the specified block type
            or if the configuration parameters are missing or incompatible.

    Examples:
        >>> input_config = {"vgg_like": True, "subsampling_factor": 4, "conv_size": 64}
        >>> body_config = {"block_type": "conformer", "hidden_size": 128}
        >>> output_size = validate_input_block(input_config, body_config, 256)
        >>> print(output_size)
        128

    Note:
        - The function supports both VGG-like and standard Conv2D input modules.
        - Valid subsampling factors differ based on the input module type.
    """
    vgg_like = configuration.get("vgg_like", False)
    next_block_type = body_first_conf.get("block_type")
    allowed_next_block_type = ["branchformer", "conformer", "conv1d", "ebranchformer"]

    if next_block_type is None or (next_block_type not in allowed_next_block_type):
        return -1

    if configuration.get("subsampling_factor") is None:
        configuration["subsampling_factor"] = 4
    sub_factor = configuration["subsampling_factor"]

    if vgg_like:
        conv_size = configuration.get("conv_size", (64, 128))

        if isinstance(conv_size, int):
            conv_size = (conv_size, conv_size)

        if sub_factor not in [4, 6]:
            raise ValueError(
                "VGG2L input module only support subsampling factor of 4 and 6."
            )
    else:
        conv_size = configuration.get("conv_size", None)

        if isinstance(conv_size, tuple):
            conv_size = conv_size[0]

        if sub_factor not in [2, 4, 6]:
            raise ValueError(
                "Conv2D input module only support subsampling factor of 2, 4 and 6."
            )

    if next_block_type == "conv1d":
        if vgg_like:
            _, output_size = get_convinput_module_parameters(
                input_size, conv_size[1], sub_factor, is_vgg=True
            )
        else:
            if conv_size is None:
                conv_size = body_first_conf.get("output_size", 64)

            _, output_size = get_convinput_module_parameters(
                input_size, conv_size, sub_factor, is_vgg=False
            )

        configuration["output_size"] = None
    else:
        output_size = body_first_conf.get("hidden_size")

        if conv_size is None:
            conv_size = output_size

        configuration["output_size"] = output_size

    configuration["conv_size"] = conv_size
    configuration["vgg_like"] = vgg_like

    return output_size


def validate_architecture(
    input_conf: Dict[str, Any], body_conf: List[Dict[str, Any]], input_size: int
) -> Tuple[int, int]:
    """
    Validate specified architecture is valid.

    This function ensures that the input block and body blocks of the encoder 
    architecture conform to the required configurations and constraints. It checks 
    the input sizes and validates that the output of one block matches the input 
    of the subsequent block.

    Args:
        input_conf: A dictionary containing the configuration for the encoder 
            input block.
        body_conf: A list of dictionaries containing the configurations for the 
            encoder body blocks.
        input_size: An integer representing the size of the encoder input.

    Returns:
        Tuple[int, int]: A tuple containing the output size of the encoder input 
        block and the output size of the encoder body block.

    Raises:
        ValueError: If there is an output/input size mismatch between blocks or 
        if the configurations are invalid.

    Examples:
        >>> input_config = {"vgg_like": True, "subsampling_factor": 4}
        >>> body_config = [{"block_type": "conv1d", "hidden_size": 128}]
        >>> validate_architecture(input_config, body_config, 64)
        (64, 128)

    Note:
        The function relies on `validate_input_block` and `validate_block_arguments` 
        to perform the necessary validations on the blocks.
    """
    input_block_osize = validate_input_block(input_conf, body_conf[0], input_size)

    cmp_io = []

    for i, b in enumerate(body_conf):
        _io = validate_block_arguments(
            b, (i + 1), input_block_osize if i == 0 else cmp_io[i - 1][1]
        )

        cmp_io.append(_io)

    for i in range(1, len(cmp_io)):
        if cmp_io[(i - 1)][1] != cmp_io[i][0]:
            raise ValueError(
                "Output/Input mismatch between blocks %d and %d"
                " in the encoder body." % ((i - 1), i)
            )

    return input_block_osize, cmp_io[-1][1]
