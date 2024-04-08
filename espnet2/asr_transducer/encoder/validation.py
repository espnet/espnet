"""Set of methods to validate encoder architecture."""

from typing import Any, Dict, List, Tuple

from espnet2.asr_transducer.utils import get_convinput_module_parameters


def validate_block_arguments(
    configuration: Dict[str, Any],
    block_id: int,
    previous_block_output: int,
) -> Tuple[int, int]:
    """Validate block arguments.

    Args:
        configuration: Architecture configuration.
        block_id: Block ID.
        previous_block_output: Previous block output size.

    Returns:
        input_size: Block input size.
        output_size: Block output size.

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
    """Validate input block.

    Args:
        configuration: Encoder input block configuration.
        body_first_conf: Encoder first body block configuration.
        input_size: Encoder input block input size.

    Return:
        output_size: Encoder input block output size.

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
    """Validate specified architecture is valid.

    Args:
        input_conf: Encoder input block configuration.
        body_conf: Encoder body blocks configuration.
        input_size: Encoder input size.

    Returns:
        input_block_osize: Encoder input block output size.
        : Encoder body block output size.

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
