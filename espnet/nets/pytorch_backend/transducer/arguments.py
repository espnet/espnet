"""Transducer model arguments."""

import ast
from argparse import _ArgumentGroup
from distutils.util import strtobool


def add_encoder_general_arguments(group: _ArgumentGroup) -> _ArgumentGroup:
    """Define general arguments for encoder."""
    group.add_argument(
        "--etype",
        default="blstmp",
        type=str,
        choices=[
            "custom",
            "lstm",
            "blstm",
            "lstmp",
            "blstmp",
            "vgglstmp",
            "vggblstmp",
            "vgglstm",
            "vggblstm",
            "gru",
            "bgru",
            "grup",
            "bgrup",
            "vgggrup",
            "vggbgrup",
            "vgggru",
            "vggbgru",
        ],
        help="Type of encoder network architecture",
    )
    group.add_argument(
        "--dropout-rate", default=0.0, type=float, help="Dropout rate for the encoder",
    )

    return group


def add_rnn_encoder_arguments(group: _ArgumentGroup) -> _ArgumentGroup:
    """Define arguments for RNN encoder."""
    group.add_argument(
        "--elayers",
        default=4,
        type=int,
        help="Number of encoder layers (for shared recognition part "
        "in multi-speaker asr mode)",
    )
    group.add_argument(
        "--eunits", "-u", default=300, type=int, help="Number of encoder hidden units",
    )
    group.add_argument(
        "--eprojs", default=320, type=int, help="Number of encoder projection units"
    )
    group.add_argument(
        "--subsample",
        default="1",
        type=str,
        help="Subsample input frames x_y_z means subsample every x frame "
        "at 1st layer, every y frame at 2nd layer etc.",
    )

    return group


def add_custom_encoder_arguments(group: _ArgumentGroup) -> _ArgumentGroup:
    """Define arguments for Custom encoder."""
    group.add_argument(
        "--enc-block-arch",
        type=eval,
        action="append",
        default=None,
        help="Encoder architecture definition by blocks",
    )
    group.add_argument(
        "--enc-block-repeat",
        default=1,
        type=int,
        help="Repeat N times the provided encoder blocks if N > 1",
    )
    group.add_argument(
        "--custom-enc-input-layer",
        type=str,
        default="conv2d",
        choices=["conv2d", "vgg2l", "linear", "embed"],
        help="Custom encoder input layer type",
    )
    group.add_argument(
        "--custom-enc-input-dropout-rate",
        type=float,
        default=0.0,
        help="Dropout rate of custom encoder input layer",
    )
    group.add_argument(
        "--custom-enc-input-pos-enc-dropout-rate",
        type=float,
        default=0.0,
        help="Dropout rate of positional encoding in custom encoder input layer",
    )
    group.add_argument(
        "--custom-enc-positional-encoding-type",
        type=str,
        default="abs_pos",
        choices=["abs_pos", "scaled_abs_pos", "rel_pos"],
        help="Custom encoder positional encoding layer type",
    )
    group.add_argument(
        "--custom-enc-self-attn-type",
        type=str,
        default="self_attn",
        choices=["self_attn", "rel_self_attn"],
        help="Custom encoder self-attention type",
    )
    group.add_argument(
        "--custom-enc-pw-activation-type",
        type=str,
        default="relu",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Custom encoder pointwise activation type",
    )
    group.add_argument(
        "--custom-enc-conv-mod-activation-type",
        type=str,
        default="swish",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Custom encoder convolutional module activation type",
    )

    return group


def add_decoder_general_arguments(group: _ArgumentGroup) -> _ArgumentGroup:
    """Define general arguments for encoder."""
    group.add_argument(
        "--dtype",
        default="lstm",
        type=str,
        choices=["lstm", "gru", "custom"],
        help="Type of decoder to use",
    )
    group.add_argument(
        "--dropout-rate-decoder",
        default=0.0,
        type=float,
        help="Dropout rate for the decoder",
    )
    group.add_argument(
        "--dropout-rate-embed-decoder",
        default=0.0,
        type=float,
        help="Dropout rate for the decoder embedding layer",
    )

    return group


def add_rnn_decoder_arguments(group: _ArgumentGroup) -> _ArgumentGroup:
    """Define arguments for RNN decoder."""
    group.add_argument(
        "--dec-embed-dim",
        default=320,
        type=int,
        help="Number of decoder embeddings dimensions",
    )
    group.add_argument(
        "--dlayers", default=1, type=int, help="Number of decoder layers"
    )
    group.add_argument(
        "--dunits", default=320, type=int, help="Number of decoder hidden units"
    )

    return group


def add_custom_decoder_arguments(group: _ArgumentGroup) -> _ArgumentGroup:
    """Define arguments for Custom decoder."""
    group.add_argument(
        "--dec-block-arch",
        type=eval,
        action="append",
        default=None,
        help="Custom decoder blocks definition",
    )
    group.add_argument(
        "--dec-block-repeat",
        default=1,
        type=int,
        help="Repeat N times the provided decoder blocks if N > 1",
    )
    group.add_argument(
        "--custom-dec-input-layer",
        type=str,
        default="embed",
        choices=["linear", "embed"],
        help="Custom decoder input layer type",
    )
    group.add_argument(
        "--custom-dec-pw-activation-type",
        type=str,
        default="relu",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Custom decoder pointwise activation type",
    )

    return group


def add_custom_training_arguments(group: _ArgumentGroup) -> _ArgumentGroup:
    """Define arguments for training with Custom architecture."""
    group.add_argument(
        "--optimizer-warmup-steps",
        default=25000,
        type=int,
        help="Optimizer warmup steps",
    )
    group.add_argument(
        "--noam-lr", default=10.0, type=float, help="Initial value of learning rate",
    )
    group.add_argument(
        "--noam-adim",
        default=0,
        type=int,
        help="Most dominant attention dimension for scheduler.",
    )
    group.add_argument(
        "--transformer-warmup-steps",
        type=int,
        help="Optimizer warmup steps. The parameter is deprecated, "
        "please use --optimizer-warmup-steps instead.",
        dest="optimizer_warmup_steps",
    )
    group.add_argument(
        "--transformer-lr",
        type=float,
        help="Initial value of learning rate. The parameter is deprecated, "
        "please use --noam-lr instead.",
        dest="noam_lr",
    )
    group.add_argument(
        "--adim",
        type=int,
        help="Most dominant attention dimension for scheduler. "
        "The parameter is deprecated, please use --noam-adim instead.",
        dest="noam_adim",
    )

    return group


def add_transducer_arguments(group: _ArgumentGroup) -> _ArgumentGroup:
    """Define general arguments for Transducer model."""
    group.add_argument(
        "--transducer-weight",
        default=1.0,
        type=float,
        help="Weight of main Transducer loss.",
    )
    group.add_argument(
        "--joint-dim",
        default=320,
        type=int,
        help="Number of dimensions in joint space",
    )
    group.add_argument(
        "--joint-activation-type",
        type=str,
        default="tanh",
        choices=["relu", "tanh", "swish"],
        help="Joint network activation type",
    )
    group.add_argument(
        "--score-norm",
        type=strtobool,
        nargs="?",
        default=True,
        help="Normalize Transducer scores by length",
    )
    group.add_argument(
        "--fastemit-lambda",
        default=0.0,
        type=float,
        help="Regularization parameter for FastEmit (https://arxiv.org/abs/2010.11148)",
    )

    return group


def add_auxiliary_task_arguments(group: _ArgumentGroup) -> _ArgumentGroup:
    """Add arguments for auxiliary task."""
    group.add_argument(
        "--use-ctc-loss",
        type=strtobool,
        nargs="?",
        default=False,
        help="Whether to compute auxiliary CTC loss.",
    )
    group.add_argument(
        "--ctc-loss-weight",
        default=0.5,
        type=float,
        help="Weight of auxiliary CTC loss.",
    )
    group.add_argument(
        "--ctc-loss-dropout-rate",
        default=0.0,
        type=float,
        help="Dropout rate for auxiliary CTC.",
    )
    group.add_argument(
        "--use-lm-loss",
        type=strtobool,
        nargs="?",
        default=False,
        help="Whether to compute auxiliary LM loss (label smoothing).",
    )
    group.add_argument(
        "--lm-loss-weight",
        default=0.5,
        type=float,
        help="Weight of auxiliary LM loss.",
    )
    group.add_argument(
        "--lm-loss-smoothing-rate",
        default=0.0,
        type=float,
        help="Smoothing rate for LM loss. If > 0, label smoothing is enabled.",
    )
    group.add_argument(
        "--use-aux-transducer-loss",
        type=strtobool,
        nargs="?",
        default=False,
        help="Whether to compute auxiliary Transducer loss.",
    )
    group.add_argument(
        "--aux-transducer-loss-weight",
        default=0.2,
        type=float,
        help="Weight of auxiliary Transducer loss.",
    )
    group.add_argument(
        "--aux-transducer-loss-enc-output-layers",
        default=None,
        type=ast.literal_eval,
        help="List of intermediate encoder layers for auxiliary "
        "transducer loss computation.",
    )
    group.add_argument(
        "--aux-transducer-loss-mlp-dim",
        default=320,
        type=int,
        help="Multilayer perceptron hidden dimension for auxiliary Transducer loss.",
    )
    group.add_argument(
        "--aux-transducer-loss-mlp-dropout-rate",
        default=0.0,
        type=float,
        help="Multilayer perceptron dropout rate for auxiliary Transducer loss.",
    )
    group.add_argument(
        "--use-symm-kl-div-loss",
        type=strtobool,
        nargs="?",
        default=False,
        help="Whether to compute symmetric KL divergence loss.",
    )
    group.add_argument(
        "--symm-kl-div-loss-weight",
        default=0.2,
        type=float,
        help="Weight of symmetric KL divergence loss.",
    )

    return group
