# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.e2e_st_md_transformer import E2E as E2EMdTransformer
from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,  # noqa:H301
    verify_rel_pos_type,  # noqa: H301
)


class E2E(E2EMdTransformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2EMdTransformer.add_arguments(parser)
        E2E.add_conformer_arguments(parser)
        return parser

    @staticmethod
    def add_conformer_arguments(parser):
        """Add arguments for conformer model."""
        group = parser.add_argument_group("conformer model specific setting")
        group = add_arguments_conformer_common(group)
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1, odim_si=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(idim, odim, args, ignore_id, odim_si)

        # Check the relative positional encoding type
        args = verify_rel_pos_type(args)

        self.encoder_asr = Encoder(
            idim=idim,
            attention_dim=args.enc_inp_adim,
            attention_heads=args.enc_inp_aheads,
            linear_units=args.enc_inp_units,
            num_blocks=args.enc_inp_layers,
            input_layer=args.enc_inp_input_layer,
            dropout_rate=args.enc_inp_dropout_rate,
            positional_dropout_rate=args.enc_inp_dropout_rate,
            attention_dropout_rate=args.enc_inp_transformer_attn_dropout_rate,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
        )

        self.reset_parameters(args)
        if args.init_like_bert_enc:
            self.init_like_bert_enc()
        if args.init_like_bert_dec:
            self.init_like_bert_dec()
