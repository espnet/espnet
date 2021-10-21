# Copyright 2020 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer common arguments."""


from distutils.util import strtobool


def add_arguments_transformer_common(group):
    """Add Transformer common arguments."""
    group.add_argument(
        "--transformer-init",
        type=str,
        default="pytorch",
        choices=[
            "pytorch",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
        ],
        help="how to initialize transformer parameters",
    )
    group.add_argument(
        "--transformer-input-layer",
        type=str,
        default="conv2d",
        choices=["conv2d", "linear", "embed"],
        help="transformer input layer type",
    )
    group.add_argument(
        "--transformer-attn-dropout-rate",
        default=None,
        type=float,
        help="dropout in transformer attention. use --dropout-rate if None is set",
    )
    group.add_argument(
        "--transformer-lr",
        default=10.0,
        type=float,
        help="Initial value of learning rate",
    )
    group.add_argument(
        "--transformer-warmup-steps",
        default=25000,
        type=int,
        help="optimizer warmup steps",
    )
    group.add_argument(
        "--transformer-length-normalized-loss",
        default=True,
        type=strtobool,
        help="normalize loss by length",
    )
    group.add_argument(
        "--transformer-encoder-selfattn-layer-type",
        type=str,
        default="selfattn",
        choices=[
            "selfattn",
            "rel_selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer encoder self-attention layer type",
    )
    group.add_argument(
        "--transformer-decoder-selfattn-layer-type",
        type=str,
        default="selfattn",
        choices=[
            "selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer decoder self-attention layer type",
    )
    # Lightweight/Dynamic convolution related parameters.
    # See https://arxiv.org/abs/1912.11793v2
    # and https://arxiv.org/abs/1901.10430 for detail of the method.
    # Configurations used in the first paper are in
    # egs/{csj, librispeech}/asr1/conf/tuning/ld_conv/
    group.add_argument(
        "--wshare",
        default=4,
        type=int,
        help="Number of parameter shargin for lightweight convolution",
    )
    group.add_argument(
        "--ldconv-encoder-kernel-length",
        default="21_23_25_27_29_31_33_35_37_39_41_43",
        type=str,
        help="kernel size for lightweight/dynamic convolution: "
        'Encoder side. For example, "21_23_25" means kernel length 21 for '
        "First layer, 23 for Second layer and so on.",
    )
    group.add_argument(
        "--ldconv-decoder-kernel-length",
        default="11_13_15_17_19_21",
        type=str,
        help="kernel size for lightweight/dynamic convolution: "
        'Decoder side. For example, "21_23_25" means kernel length 21 for '
        "First layer, 23 for Second layer and so on.",
    )
    group.add_argument(
        "--ldconv-usebias",
        type=strtobool,
        default=False,
        help="use bias term in lightweight/dynamic convolution",
    )
    group.add_argument(
        "--dropout-rate", default=0.0, type=float, help="Dropout rate for the encoder"
    )
    group.add_argument(
        "--intermediate-ctc-weight",
        default=0.0,
        type=float,
        help="Weight of intermediate CTC weight",
    )
    group.add_argument(
        "--intermediate-ctc-layer",
        default="",
        type=str,
        help="Position of intermediate CTC layer. {int} or {int},{int},...,{int}",
    )
    # Encoder
    group.add_argument(
        "--elayers",
        default=4,
        type=int,
        help="Number of encoder layers (for shared recognition part "
        "in multi-speaker asr mode)",
    )
    group.add_argument(
        "--eunits", "-u", default=300, type=int, help="Number of encoder hidden units"
    )
    # Attention
    group.add_argument(
        "--adim",
        default=320,
        type=int,
        help="Number of attention transformation dimensions",
    )
    group.add_argument(
        "--aheads", default=4, type=int, help="Number of heads for multi head attention"
    )
    group.add_argument(
        "--stochastic-depth-rate",
        default=0.0,
        type=float,
        help="Skip probability of stochastic layer regularization",
    )
    # Decoder
    group.add_argument(
        "--dlayers", default=1, type=int, help="Number of decoder layers"
    )
    group.add_argument(
        "--dunits", default=320, type=int, help="Number of decoder hidden units"
    )
    return group


def add_arguments_md_transformer_common(group):
    """Add Basic Transformer arguments."""
    ######
    group.add_argument(
        "--transformer-init",
        type=str,
        default="pytorch",
        choices=[
            "pytorch",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
        ],
        help="how to initialize transformer parameters",
    )
    group.add_argument(
        "--transformer-lr",
        default=10.0,
        type=float,
        help="Initial value of learning rate",
    )
    group.add_argument(
        "--transformer-warmup-steps",
        default=25000,
        type=int,
        help="optimizer warmup steps",
    )
    group.add_argument(
        "--transformer-length-normalized-loss",
        default=True,
        type=strtobool,
        help="normalize loss by length",
    )
    group.add_argument(
        "--transformer-encoder-selfattn-layer-type",
        type=str,
        default="selfattn",
        choices=[
            "selfattn",
            "rel_selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer encoder self-attention layer type",
    )
    ######
    # Lightweight/Dynamic convolution related parameters.
    # See https://arxiv.org/abs/1912.11793v2
    # and https://arxiv.org/abs/1901.10430 for detail of the method.
    # Configurations used in the first paper are in
    # egs/{csj, librispeech}/asr1/conf/tuning/ld_conv/
    group.add_argument(
        "--wshare",
        default=4,
        type=int,
        help="Number of parameter shargin for lightweight convolution",
    )
    group.add_argument(
        "--ldconv-encoder-kernel-length",
        default="21_23_25_27_29_31_33_35_37_39_41_43",
        type=str,
        help="kernel size for lightweight/dynamic convolution: "
        'Encoder side. For example, "21_23_25" means kernel length 21 for '
        "First layer, 23 for Second layer and so on.",
    )
    group.add_argument(
        "--ldconv-decoder-kernel-length",
        default="11_13_15_17_19_21",
        type=str,
        help="kernel size for lightweight/dynamic convolution: "
        'Decoder side. For example, "21_23_25" means kernel length 21 for '
        "First layer, 23 for Second layer and so on.",
    )
    group.add_argument(
        "--ldconv-usebias",
        type=strtobool,
        default=False,
        help="use bias term in lightweight/dynamic convolution",
    )
    group.add_argument(
        "--dropout-rate",
        default=0.0,
        type=float,
        help="Dropout rate for the entire model",
    )
    ######
    # Encoder for the Input
    group.add_argument(
        "--enc-inp-layers",
        "--elayers",
        default=4,
        type=int,
        help="Number of input encoder layers",
    )
    group.add_argument(
        "--enc-inp-units",
        "--eunits",
        default=2048,
        type=int,
        help="Number of encoder hidden units",
    )
    group.add_argument(
        "--enc-inp-input-layer",
        "--transformer-input-layer",
        type=str,
        default="conv2d",
        choices=["conv2d", "linear", "embed"],
        help="transformer input layer type",
    )
    group.add_argument(
        "--enc-inp-aheads",
        "--aheads",
        default=4,
        type=int,
        help="Number of heads for multi head attention",
    )
    group.add_argument(
        "--enc-inp-adim",
        "--adim",
        default=256,
        type=int,
        help="Number of heads for attention dim",
    )
    group.add_argument(
        "--enc-inp-dropout-rate",
        default=None,
        type=float,
        help="dropout rate for Encoder Inp. use --dropout-rate if None is set",
    )
    group.add_argument(
        "--enc-inp-transformer-attn-dropout-rate",
        "--transformer-attn-dropout-rate",
        default=None,
        type=float,
        help="Dropout in transformer attention for Enc Inp. use --dropout-rate if None is set",
    )
    # Decoder for the Output
    group.add_argument(
        "--dec-out-layers",
        "--dlayers",
        default=6,
        type=int,
        help="Number of decoder layers",
    )
    group.add_argument(
        "--dec-out-units",
        "--dunits",
        default=2048,
        type=int,
        help="Number of decoder hidden units",
    )
    group.add_argument(
        "--dec-out-transformer-selfattn-layer-type",
        type=str,
        default=None,
        choices=[
            "seqspeechattn",
            "selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer decoder self-attention layer type. use --transformer-decoder-selfattn-layer-type if None is set",
    )
    group.add_argument(
        "--dec-out-aheads",
        default=None,
        type=int,
        help="Number of heads for multi head attention. use --aheads if None is set",
    )
    group.add_argument(
        "--dec-out-adim",
        default=None,
        type=int,
        help="Number of heads for attention dim. use --adim if None is set",
    )
    group.add_argument(
        "--dec-out-dropout-rate",
        default=None,
        type=float,
        help="dropout rate for Encoder SI. use --dropout-rate if None is set",
    )
    group.add_argument(
        "--dec-out-transformer-attn-dropout-rate",
        default=None,
        type=float,
        help="Dropout in transformer attention for Enc SI. use --transformer-attn-dropout-rate if None is set",
    )

    # Decoder for the Searchable Intermediates
    group.add_argument(
        "--dec-si-layers", default=6, type=int, help="Number of decoder layers"
    )
    group.add_argument(
        "--dec-si-units",
        default=None,
        type=int,
        help="Number of decoder hidden units. use --dunits if None is set",
    )
    group.add_argument(
        "--dec-si-transformer-selfattn-layer-type",
        "--transformer-decoder-selfattn-layer-type",
        type=str,
        default="selfattn",
        choices=[
            "selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer decoder self-attention layer type",
    )
    group.add_argument(
        "--dec-si-aheads",
        default=None,
        type=int,
        help="Number of heads for multi head attention. use --aheads if None is set",
    )
    group.add_argument(
        "--dec-si-adim",
        default=None,
        type=int,
        help="Number of heads for attention dim. use --adim if None is set",
    )
    group.add_argument(
        "--dec-si-dropout-rate",
        default=None,
        type=float,
        help="dropout rate for Encoder SI. use --dec-out-dropout-rate if None is set",
    )
    group.add_argument(
        "--dec-si-transformer-attn-dropout-rate",
        default=None,
        type=float,
        help="Dropout in transformer attention for Enc SI. use --dec-out-transformer-attn-dropout-rate if None is set",
    )

    # Encoder for the Encoding the Searchable Intermediates
    group.add_argument(
        "--enc-si-layers", default=2, type=int, help="Number of encoder layers for SI"
    )
    group.add_argument(
        "--enc-si-input-layer",
        type=str,
        default="nothing",
        help="Input layer for encoder SI",
    )
    group.add_argument(
        "--enc-si-units",
        default=None,
        type=int,
        help="Number of encoder SI hidden units. use --eunits if None is set",
    )
    group.add_argument(
        "--enc-si-aheads",
        default=None,
        type=int,
        help="Number of heads for multi head attention. use --aheads if None is set",
    )
    group.add_argument(
        "--enc-si-adim",
        default=None,
        type=int,
        help="Number of heads for attention dim. use --adim if None is set",
    )
    group.add_argument(
        "--enc-si-dropout-rate",
        default=None,
        type=float,
        help="dropout rate for Encoder SI. use --enc-inp-dropout-rate if None is set",
    )
    group.add_argument(
        "--enc-si-transformer-attn-dropout-rate",
        default=None,
        type=float,
        help="Dropout in transformer attention for Enc SI. use --enc-inp-transformer-attn-dropout-rate if None is set",
    )
    group.add_argument(
        "--enc-si-transformer-selfattn-layer-type",
        type=str,
        default="selfattn",
        choices=[
            "selfattn",
            "rel_selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer encoder self-attention layer type",
    )
    ######
    return group
