#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import torch

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import make_non_pad_mask
from espnet.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet.nets.pytorch_backend.e2e_tts_transformer import TTSPlot
from espnet.nets.pytorch_backend.fastspeech.duration_calculator import DurationCalculator
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictorLoss
from espnet.nets.pytorch_backend.fastspeech.length_regularizer import LengthRegularizer
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool


class FeedForwardTransformer(TTSInterface, torch.nn.Module):
    """Feed Forward Transformer for TTS a.k.a. FastSpeech

    Reference:
        FastSpeech: Fast, Robust and Controllable Text to Speech
        (https://arxiv.org/pdf/1905.09263.pdf)

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param Namespace args: argments containing following attributes
        (int) adim: number of attention transformation dimensions
        (int) aheads: number of heads for multi head attention
        (int) elayers: number of encoder layers
        (int) eunits: number of encoder hidden units
        (int) dlayers: number of decoder layers
        (int) dunits: number of decoder hidden units
        (str) positionwise_layer_type: layer type in positionwise operation
        (str) positionwise_conv_kernel_size: kernel size of positionwise conv layer
        (bool) use_scaled_pos_enc: whether to use trainable scaled positional encoding instead of the fixed scale one
        (bool) encoder_normalize_before: whether to perform layer normalization before encoder block
        (bool) decoder_normalize_before: whether to perform layer normalization before decoder block
        (bool) encoder_concat_after: whether to concatenate attention layer's input and output in encoder
        (bool) decoder_concat_after: whether to concatenate attention layer's input and output in decoder
        (int) duration_predictor_layers: number of duration predictor layers
        (int) duration_predictor_chans: number of duration predictor channels
        (int) duration_predictor_kernel_size: kernel size of duration predictor
        (str) teacher_model: teacher auto-regressive transformer model path
        (int) reduction_factor: reduction factor
        (float) transformer_init: how to initialize transformer parameters
        (float) transformer_lr: initial value of learning rate
        (int) transformer_warmup_steps: optimizer warmup steps
        (float) transformer_enc_dropout_rate: dropout rate in encoder except for attention and positional encoding
        (float) transformer_enc_positional_dropout_rate: dropout rate after encoder positional encoding
        (float) transformer_enc_attn_dropout_rate: dropout rate in encoder self-attention module
        (float) transformer_dec_dropout_rate: dropout rate in decoder except for attention and positional encoding
        (float) transformer_dec_positional_dropout_rate:  dropout rate after decoder positional encoding
        (float) transformer_dec_attn_dropout_rate: dropout rate in deocoder self-attention module
        (float) transformer_enc_dec_attn_dropout_rate: dropout rate in encoder-deocoder attention module
        (bool) init_encoder_from_teacher: whether to initialize encoder using teacher encoder parameters
        (str) init_encoder_module: encoder module to be initialized using teacher parameters
        (bool) use_masking: whether to use masking in calculation of loss
    """

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("feed-forward transformer model setting")
        # network structure related
        group.add_argument("--adim", default=384, type=int,
                           help="Number of attention transformation dimensions")
        group.add_argument("--aheads", default=4, type=int,
                           help="Number of heads for multi head attention")
        group.add_argument("--elayers", default=6, type=int,
                           help="Number of encoder layers")
        group.add_argument("--eunits", default=1536, type=int,
                           help="Number of encoder hidden units")
        group.add_argument("--dlayers", default=6, type=int,
                           help="Number of decoder layers")
        group.add_argument("--dunits", default=1536, type=int,
                           help="Number of decoder hidden units")
        group.add_argument("--positionwise-layer-type", default="linear", type=str,
                           choices=["linear", "conv1d"],
                           help="Positionwise layer type.")
        group.add_argument("--positionwise-conv-kernel-size", default=3, type=int,
                           help="Kernel size of positionwise conv1d layer")
        group.add_argument("--use-scaled-pos-enc", default=True, type=strtobool,
                           help="Use trainable scaled positional encoding instead of the fixed scale one")
        group.add_argument("--encoder-normalize-before", default=False, type=strtobool,
                           help="Whether to apply layer norm before encoder block")
        group.add_argument("--decoder-normalize-before", default=False, type=strtobool,
                           help="Whether to apply layer norm before decoder block")
        group.add_argument("--encoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer's input and output in encoder")
        group.add_argument("--decoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer's input and output in decoder")
        group.add_argument("--duration-predictor-layers", default=2, type=int,
                           help="Number of layers in duration predictor")
        group.add_argument("--duration-predictor-chans", default=384, type=int,
                           help="Number of channels in duration predictor")
        group.add_argument("--duration-predictor-kernel-size", default=3, type=int,
                           help="Kernel size in duration predictor")
        group.add_argument("--teacher-model", default=None, type=str, nargs="?",
                           help="Teacher model file path")
        parser.add_argument("--reduction-factor", default=1, type=int,
                            help="Reduction factor")
        # training related
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help="How to initialize transformer parameters")
        group.add_argument("--initial-encoder-alpha", type=float, default=1.0,
                           help="Initial alpha value in encoder's ScaledPositionalEncoding")
        group.add_argument("--initial-decoder-alpha", type=float, default=1.0,
                           help="Initial alpha value in decoder's ScaledPositionalEncoding")
        group.add_argument("--transformer-lr", default=1.0, type=float,
                           help="Initial value of learning rate")
        group.add_argument("--transformer-warmup-steps", default=4000, type=int,
                           help="Optimizer warmup steps")
        group.add_argument("--transformer-enc-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder except for attention")
        group.add_argument("--transformer-enc-positional-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder positional encoding")
        group.add_argument("--transformer-enc-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder self-attention")
        group.add_argument("--transformer-dec-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer decoder except for attention and pos encoding")
        group.add_argument("--transformer-dec-positional-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer decoder positional encoding")
        group.add_argument("--transformer-dec-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer decoder self-attention")
        group.add_argument("--transformer-enc-dec-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder-decoder attention")
        group.add_argument("--duration-predictor-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for duration predictor")
        group.add_argument("--init-encoder-from-teacher", default=True, type=strtobool,
                           help="Whether to initialize encoder using teacher's parameters")
        group.add_argument("--init-encoder-module", default="all", type=str,
                           choices=["all", "embed"],
                           help="Encoder modeules to be initilized")
        # loss related
        group.add_argument("--use-masking", default=True, type=strtobool,
                           help="Whether to use masking in calculation of loss")

        return parser

    def __init__(self, idim, odim, args):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.reduction_factor = args.reduction_factor
        self.use_scaled_pos_enc = args.use_scaled_pos_enc
        self.use_masking = args.use_masking

        # TODO(kan-bayashi): support reduction_factor > 1
        if self.reduction_factor != 1:
            raise NotImplementedError("Support only reduction_factor = 1.")

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim,
            embedding_dim=args.adim,
            padding_idx=padding_idx
        )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=args.transformer_enc_dropout_rate,
            positional_dropout_rate=args.transformer_enc_positional_dropout_rate,
            attention_dropout_rate=args.transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.encoder_normalize_before,
            concat_after=args.encoder_concat_after,
            positionwise_layer_type=args.positionwise_layer_type,
            positionwise_conv_kernel_size=args.positionwise_conv_kernel_size
        )

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=args.adim,
            n_layers=args.duration_predictor_layers,
            n_chans=args.duration_predictor_chans,
            kernel_size=args.duration_predictor_kernel_size,
            dropout_rate=args.duration_predictor_dropout_rate,
        )

        # define length regularizer
        self.length_regularizer = LengthRegularizer()

        # define decoder
        self.decoder = Encoder(
            idim=0,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            input_layer=None,
            dropout_rate=args.transformer_dec_dropout_rate,
            positional_dropout_rate=args.transformer_dec_positional_dropout_rate,
            attention_dropout_rate=args.transformer_dec_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.decoder_normalize_before,
            concat_after=args.decoder_concat_after,
            positionwise_layer_type=args.positionwise_layer_type,
            positionwise_conv_kernel_size=args.positionwise_conv_kernel_size
        )

        # define final projection
        self.feat_out = torch.nn.Linear(args.adim, odim * args.reduction_factor)

        # initialize parameters
        self._reset_parameters(args)

        # define teacher model
        if args.teacher_model is not None:
            self.teacher = self._load_teacher_model(args.teacher_model)
        else:
            self.teacher = None

        # define duration calculator
        if self.teacher is not None:
            self.duration_calculator = DurationCalculator(self.teacher)
        else:
            self.duration_calculator = None

        # transfer teacher encoder parameters
        if args.init_encoder_from_teacher:
            self._init_encoder_from_teacher(args.init_encoder_module)

        # define criterions
        self.duration_criterion = DurationPredictorLoss()
        # TODO(kan-bayashi): support knowledge distillation loss
        self.criterion = torch.nn.L1Loss()

    def forward(self, xs, ilens, ys, labels, olens, *args, **kwargs):
        """Transformer forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor olens: batch of the lengths of each target (B)
        :return: loss value
        :rtype: torch.Tensor
        """
        # remove unnecessary padded part (for multi-gpus)
        max_ilen = max(ilens)
        max_olen = max(olens)
        if max_ilen != xs.shape[1]:
            xs = xs[:, :max_ilen]
        if max_olen != ys.shape[1]:
            ys = ys[:, :max_olen]

        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        # calculate groundtruth duration
        with torch.no_grad():
            ds = self.duration_calculator(xs, ilens, ys, olens)  # (B, Tmax)

        # calculate predicted duration
        d_masks = make_pad_mask(ilens).to(xs.device)
        d_outs = self.duration_predictor(hs, d_masks)  # (B, Tmax)

        # apply length regularizer
        hs = self.length_regularizer(hs, ds, ilens)  # (B, Lmax, adim)

        # forward decoder
        h_masks = self._source_mask(olens)
        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        # apply mask to remove padded part
        if self.use_masking:
            y_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            outs = outs.masked_select(y_masks)
            ys = ys.masked_select(y_masks)
            d_outs = d_outs.masked_select(~d_masks)
            ds = ds.masked_select(~d_masks)

        # calculate loss
        l1_loss = self.criterion(outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        loss = l1_loss + duration_loss
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"duration_loss": duration_loss.item()},
            {"loss": loss.item()},
        ]

        # report extra information
        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed[-1].alpha.data.item()},
            ]
        self.reporter.report(report_keys)

        return loss

    def calculate_all_attentions(self, xs, ilens, ys, olens, *args, **kwargs):
        """Calculate attention weights

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor ilens: list of lengths of each output batch (B)
        :return: attention weights dict
        :rtype: dict
        """
        with torch.no_grad():
            # remove unnecessary padded part (for multi-gpus)
            max_ilen = max(ilens)
            max_olen = max(olens)
            if max_ilen != xs.shape[1]:
                xs = xs[:, :max_ilen]
            if max_olen != ys.shape[1]:
                ys = ys[:, :max_olen]

            # forward encoder
            x_masks = self._source_mask(ilens)
            hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

            # calculate groundtruth duration
            with torch.no_grad():
                ds = self.duration_calculator(xs, ilens, ys, olens)  # (B, Tmax)

            # apply length regularizer
            hs = self.length_regularizer(hs, ds, ilens)  # (B, Lmax, adim)

            # forward decoder
            h_masks = self._source_mask(olens)
            zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
            outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        att_ws_dict = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                attn = m.attn.cpu().numpy()
                if "encoder" in name:
                    attn = [a[:, :l, :l] for a, l in zip(attn, ilens.tolist())]
                elif "decoder" in name:
                    if "src" in name:
                        attn = [a[:, :ol, :il] for a, il, ol in zip(attn, ilens.tolist(), olens.tolist())]
                    elif "self" in name:
                        attn = [a[:, :l, :l] for a, l in zip(attn, olens.tolist())]
                    else:
                        logging.warning("unknown attention module: " + name)
                else:
                    logging.warning("unknown attention module: " + name)
                att_ws_dict[name] = attn

        att_ws_dict["predicted_fbank"] = [m[:l].T for m, l in zip(outs.cpu().numpy(), olens.tolist())]
        return att_ws_dict

    def inference(self, x, *args, **kwargs):
        """Generates the sequence of features from given a sequences of characters

        :param torch.Tensor x: the sequence of character ids (T)
        :return: the sequence of generated features (1, L, odim)
        :rtype: torch.Tensor
        """
        # forward encoder
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)
        hs, _ = self.encoder(xs, None)  # (B, Tmax, adim)

        d_outs = self.duration_predictor.inference(hs, None)  # (B, Tmax)

        # apply length regularizer
        hs = self.length_regularizer(hs, d_outs, ilens)  # (B, Lmax, adim)

        # forward decoder
        zs, _ = self.decoder(hs, None)  # (B, Lmax, adim)
        outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        # keep batch axis to be compatible with the other models
        return outs

    def _source_mask(self, ilens):
        """Make mask for MultiHeadedAttention using padded sequences

        >>> ilens = [5, 3]
        >>> self._source_mask(ilens)
        tensor([[[1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1]],

                [[1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)

    def _load_teacher_model(self, model_path):
        # get teacher model config
        idim, odim, args = get_model_conf(model_path)

        # assert dimension is the same between teacher and studnet
        assert idim == self.idim
        assert odim == self.odim
        assert args.reduction_factor == self.reduction_factor

        # load teacher model
        model = Transformer(idim, odim, args)
        torch_load(model_path, model)

        # freeze teacher model parameters
        for p in model.parameters():
            p.requires_grad = False

        return model

    def _init_encoder_from_teacher(self, init_module="all"):
        if init_module == "all":
            for (n1, p1), (n2, p2) in zip(self.encoder.named_parameters(), self.teacher.encoder.named_parameters()):
                assert n1 == n2, "It seems that encoder structure is different."
                assert p1.shape == p2.shape, "It seems that encoder size is different."
                p1.data.copy_(p2.data)
        elif init_module == "embed":
            student_shape = self.encoder.embed[0].weight.data.shape
            teacher_shape = self.teacher.encoder.embed[0].weight.data.shape
            assert student_shape == teacher_shape, "It seems that embed dimension is different."
            self.encoder.embed[0].weight.data.copy_(
                self.teacher.encoder.embed[0].weight.data)
        else:
            raise NotImplementedError("Support only all or embed.")

    def _reset_parameters(self, args):
        if self.use_scaled_pos_enc:
            # alpha in scaled positional encoding init
            self.encoder.embed[-1].alpha.data = torch.tensor(args.initial_encoder_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(args.initial_decoder_alpha)

        if args.transformer_init == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if args.transformer_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif args.transformer_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif args.transformer_init == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif args.transformer_init == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + args.transformer_init)
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()

    @property
    def attention_plot_class(self):
        return TTSPlot

    @property
    def base_plot_keys(self):
        """base key names to plot during training. keys should match what `chainer.reporter` reports

        if you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        :rtype list[str] plot_keys: base keys to plot during training
        """
        plot_keys = ["loss", "l1_loss", "duration_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]

        return plot_keys
