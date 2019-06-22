#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import torch

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet.nets.pytorch_backend.e2e_tts_transformer import TTSPlot
from espnet.nets.pytorch_backend.fastspeech.duration_calculator import DurationCalculator
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictorLoss
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool


class FeedForwardTransformer(TTSInterface, torch.nn.Module):
    """Feed Forward Transformer for TTS a.k.a. FastSpeech.

    This is a module of FastSpeech, feed-forward Transformer with duration predictor described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_, which does not require any auto-regressive
    processing during inference, resulting in fast decoding compared with auto-regressive Transformer.

    Args:
        idim (int): Dimension of the inputs.
        odim (int): Dimension of the outputs.
        args (Namespace): Model hyperparameters.
            - elayers (int): Number of encoder layers.
            - eunits (int): Number of encoder hidden units.
            - adim (int): Number of attention transformation dimensions.
            - aheads (int): Number of heads for multi head attention.
            - dlayers (int): Number of decoder layers.
            - dunits (int): Number of decoder hidden units.
            - use_scaled_pos_enc (bool): Whether to use trainable scaled positional encoding.
            - encoder_normalize_before (bool): Whether to perform layer normalization before encoder block.
            - decoder_normalize_before (bool): Whether to perform layer normalization before decoder block.
            - encoder_concat_after (bool): Whether to concatenate attention layer's input and output in encoder.
            - decoder_concat_after (bool): Whether to concatenate attention layer's input and output in decoder.
            - duration_predictor_layers (int): Number of duration predictor layers.
            - duration_predictor_chans (int): Number of duration predictor channels.
            - duration_predictor_kernel_size (int): Kernel size of duration predictor.
            - teacher_model (str): Teacher auto-regressive transformer model path.
            - reduction_factor (int): Reduction factor.
            - transformer_init (float): How to initialize transformer parameters.
            - transformer_lr (float): Initial value of learning rate.
            - transformer_warmup_steps (int): Optimizer warmup steps.
            - transformer_enc_dropout_rate (float): Dropout rate in encoder except for attention & positional encoding.
            - transformer_enc_positional_dropout_rate (float): Dropout rate after encoder positional encoding.
            - transformer_enc_attn_dropout_rate (float): Dropout rate in encoder self-attention module.
            - transformer_dec_dropout_rate (float): Dropout rate in decoder except for attention & positional encoding.
            - transformer_dec_positional_dropout_rate (float): Dropout rate after decoder positional encoding.
            - transformer_dec_attn_dropout_rate (float): Dropout rate in deocoder self-attention module.
            - transformer_enc_dec_attn_dropout_rate (float): Dropout rate in encoder-deocoder attention module.
            - use_masking (bool): Whether to use masking in calculation of loss.
            - transfer_encoder_from_teacher: Whether to transfer encoder using teacher encoder parameters.
            - transferred_encoder_module: Encoder module to be initialized using teacher parameters.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    @staticmethod
    def add_arguments(parser):
        """Add model-specific arguments to the parser."""
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
        group.add_argument("--transfer-encoder-from-teacher", default=True, type=strtobool,
                           help="Whether to transfer teacher's parameters")
        group.add_argument("--transferred-encoder-module", default="all", type=str,
                           choices=["all", "embed"],
                           help="Encoder modeules to be trasferred from teacher")
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

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder because fastspeech's decoder is the same as encoder
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

        # transfer teacher parameters
        if self.teacher is not None and args.transfer_encoder_from_teacher:
            self._transfer_from_teacher(args.transferred_encoder_module)

        # define criterions
        self.duration_criterion = DurationPredictorLoss()
        # TODO(kan-bayashi): support knowledge distillation loss
        self.criterion = torch.nn.L1Loss()

    def _forward(self, xs, ilens, ys=None, olens=None, is_inference=False):
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(ilens).to(xs.device)
        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, d_outs, ilens)  # (B, Lmax, adim)
        else:
            with torch.no_grad():
                ds = self.duration_calculator(xs, ilens, ys, olens)  # (B, Tmax)
            d_outs = self.duration_predictor(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, ds, ilens)  # (B, Lmax, adim)

        # forward decoder
        if olens is not None:
            h_masks = self._source_mask(olens)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        if is_inference:
            return outs
        else:
            return outs, ds, d_outs

    def forward(self, xs, ilens, ys, olens, *args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: Loss value.

        """
        # remove unnecessary padded part (for multi-gpus)
        xs = xs[:, :max(ilens)]
        ys = ys[:, :max(olens)]

        # forward propagation
        outs, ds, d_outs = self._forward(xs, ilens, ys, olens, is_inference=False)

        # apply mask to remove padded part
        if self.use_masking:
            in_masks = make_non_pad_mask(ilens).to(xs.device)
            d_outs = d_outs.masked_select(in_masks)
            ds = ds.masked_select(in_masks)
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            outs = outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)

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
            xs = xs[:, :max(ilens)]
            ys = ys[:, :max(olens)]

            # forward propagation
            outs = self._forward(xs, ilens, ys, olens, is_inference=False)[0]

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
        """Generate the sequence of features given the sequences of characters.

        Args:
            x (Tensor): Input sequence of characters (T,).

        Returns:
            Tensor: Output sequence of features (1, L, odim).

        """
        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)

        # inference
        outs = self._forward(xs, ilens, is_inference=True)  # (1, L, odim)

        # keep batch axis to be compatible with the other models
        return outs

    def _source_mask(self, ilens):
        """Make masks for self-attention.

        Examples:
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

    def _reset_parameters(self, args):
        # initialize parameters
        initialize(self, args.transformer_init)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(args.initial_encoder_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(args.initial_decoder_alpha)

    def _transfer_from_teacher(self, transferred_encoder_module):
        if transferred_encoder_module == "all":
            for (n1, p1), (n2, p2) in zip(self.encoder.named_parameters(),
                                          self.teacher.encoder.named_parameters()):
                assert n1 == n2, "It seems that encoder structure is different."
                assert p1.shape == p2.shape, "It seems that encoder size is different."
                p1.data.copy_(p2.data)
        elif transferred_encoder_module == "embed":
            student_shape = self.encoder.embed[0].weight.data.shape
            teacher_shape = self.teacher.encoder.embed[0].weight.data.shape
            assert student_shape == teacher_shape, "It seems that embed dimension is different."
            self.encoder.embed[0].weight.data.copy_(
                self.teacher.encoder.embed[0].weight.data)
        else:
            raise NotImplementedError("Support only all or embed.")

    @property
    def attention_plot_class(self):
        """Return plot class for attention weight plot."""
        return TTSPlot

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training. keys should match what `chainer.reporter` reports.

        If you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ["loss", "l1_loss", "duration_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]

        return plot_keys
