# Copyright 2020 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer common arguments."""


from distutils.util import strtobool


def add_arguments_conformer_common(group):
    """Add Transformer common arguments."""
    group.add_argument(
        "--transformer-encoder-pos-enc-layer-type",
        type=str,
        default="abs_pos",
        choices=["abs_pos", "scaled_abs_pos", "rel_pos"],
        help="transformer encoder positional encoding layer type",
    )
    group.add_argument(
        "--transformer-encoder-activation-type",
        type=str,
        default="swish",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="transformer encoder activation function type",
    )
    group.add_argument(
        "--macaron-style",
        default=False,
        type=strtobool,
        help="Whether to use macaron style for positionwise layer",
    )
    # CNN module
    group.add_argument(
        "--use-cnn-module",
        default=False,
        type=strtobool,
        help="Use convolution module or not",
    )
    group.add_argument(
        "--cnn-module-kernel",
        default=31,
        type=int,
        help="Kernel size of convolution module.",
    )
    return group
