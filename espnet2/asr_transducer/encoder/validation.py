"""Set of methods to validate encoder architecture."""

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from espnet2.asr_transducer.utils import sub_factor_to_params


def validate_positional_information(
    configuration: Dict[str, Any], block_w_pos: List[str] = ["conformer"]
) -> Tuple[bool, float]:
    """Check architecture to define whether positional information are needed.

    Args:
        configuration: Architecture configuration.
        block_w_pos: List of block type incorporating positional information.

    Returns:
        : Whether positional information are needed.
        : Average eps value in encoder for final LayerNorm.

    """
    need_pos = any([c.get("block_type") in block_w_pos for c in configuration])

    if not need_pos:
        return False, -1

    avg_eps = [c.get("avg_eps") for c in configuration if c.get("avg_eps")]

    if avg_eps:
        avg_eps = max(set(avg_eps), key=avg_eps.count)
    else:
        avg_eps = 1e-12

    return need_pos, avg_eps


def validate_block_arguments(
    configuration: Dict[str, Any],
    num_block: int,
    previous_block_output: int,
) -> Tuple[int, int]:
    """Validate block arguments.

    Args:
        configuration: Architecture configuration
        num_block: Block ID.
        previous_block_output: Previous block output dimension.

    Returns:
        dim_input : Block input size.
        dim_output : Block output size.

    """
    block_type = configuration.get("block_type")

    if block_type is None:
        raise ValueError(
            "Block %d in encoder doesn't have a type assigned. " % num_block
        )

    if block_type == "conformer":
        dim_input = configuration.get("dim_hidden")
        dim_output = configuration.get("dim_hidden")

        if configuration.get("dim_linear") is None:
            raise ValueError(
                "Missing 'dim_linear' argument for Conformer block (ID: %d)" % num_block
            )
    elif block_type == "conv1d":
        dim_input = configuration["dim_input"] = previous_block_output
        dim_output = configuration.get("dim_output")

        if configuration.get("kernel_size") is None:
            raise ValueError(
                "Missing 'kernel_size' argument for Conv1d block (ID: %d)" % num_block
            )
    elif block_type == "rnn":
        dim_input = configuration["dim_input"] = previous_block_output
        dim_output = configuration.get("dim_output")

        if dim_output is None:
            dim_output = configuration.get("dim_hidden")
    else:
        raise ValueError("Block type: %s is not supported." % block_type)

    return block_type, (dim_input, dim_output)


def validate_input_block(
    input_conf: Dict[str, Any], body_conf: List[Dict[str, Any]], dim_input: int
):
    """Validate input block.

    Args:
        input_conf: Encoder input block configuration.
        body_conf: Encoder body blocks configuration.
        dim_input: Input dimension.

    """
    block_type = input_conf.get("block_type")
    next_block_type = body_conf[0].get("block_type")

    if block_type is None:
        block_type = input_conf["block_type"] = "vgg"
    elif block_type not in ["conv2d", "vgg"]:
        raise ValueError("Input block type can be either 'vgg' or 'conv2d'")

    if next_block_type is None or (
        next_block_type not in ["rnn", "conv1d", "conformer"]
    ):
        return -1

    if next_block_type in ["rnn", "conv1d"]:
        if block_type == "vgg":
            dim_output = 128 * ((dim_input // 2) // 2)
        else:
            dim_conv = input_conf.get("dim_conv")

            if dim_conv is None:
                if next_block_type == "rnn":
                    dim_conv = body_conf[0].get("dim_hidden")
                else:
                    dim_conv = body_conf[0].get("dim_output")

            _, _, conv_odim = sub_factor_to_params(
                input_conf.get("subsampling_factor", 4), dim_input
            )

            dim_output = conv_odim * dim_conv

            input_conf["dim_conv"] = dim_conv

        input_conf["dim_output"] = None
        input_conf["dim_pos_enc"] = dim_output
    else:
        dim_output = body_conf[0].get("dim_hidden")

        input_conf["dim_output"] = dim_output

        if block_type == "conv2d":
            input_conf["dim_conv"] = dim_output

    return dim_output


def validate_architecture(
    input_conf: Dict[str, Any], body_conf: List[Dict[str, Any]], dim_input: int
) -> Tuple[int, int]:
    """Validate specified architecture is valid.

    Args:
        input_conf: Encoder input block configuration.
        body_conf: Encoder body blocks configuration.
        dim_input: Encoder input dimension.

    Returns:
        : Encoder input block output size.
        : Encoder body block output size.

    """
    cmp_io = []
    cmp_type = []

    input_block_odim = validate_input_block(input_conf, body_conf, dim_input)

    for i, b in enumerate(body_conf):
        _type, _io = validate_block_arguments(
            b, (i + 1), input_block_odim if i == 0 else cmp_io[i - 1][1]
        )

        cmp_io.append(_io)
        cmp_type.append(_type)

    for i in range(1, len(cmp_io)):
        if cmp_io[(i - 1)][1] != cmp_io[i][0]:
            raise ValueError(
                "Output/Input mismatch between blocks %d and %d"
                " in the encoder body." % ((i - 1), i)
            )

    if all(x in cmp_type for x in ["rnn", "conformer"]):
        raise ValueError(
            "Encoder can't contain RNN and Conformer blocks simultaneously."
        )

    return cmp_io[-1][1]
