# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron2-VC related modules."""

import logging
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import \
    GuidedAttentionLoss  # noqa: H301
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import \
    Tacotron2Loss  # noqa: H301
from espnet.nets.pytorch_backend.rnn.attentions import (AttForward,
                                                        AttForwardTA, AttLoc)
from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHG, CBHGLoss
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.fill_missing_args import fill_missing_args


class Tacotron2(TTSInterface, torch.nn.Module):
    """VC Tacotron2 module for VC.

    This is a module of Tacotron2-based VC model,
    which convert the sequence of acoustic features
    into the sequence of acoustic features.
    """

    @staticmethod
    def add_arguments(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group("tacotron 2 model setting")
        # encoder
        group.add_argument(
            "--elayers", default=1, type=int, help="Number of encoder layers"
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=512,
            type=int,
            help="Number of encoder hidden units",
        )
        group.add_argument(
            "--econv-layers",
            default=3,
            type=int,
            help="Number of encoder convolution layers",
        )
        group.add_argument(
            "--econv-chans",
            default=512,
            type=int,
            help="Number of encoder convolution channels",
        )
        group.add_argument(
            "--econv-filts",
            default=5,
            type=int,
            help="Filter size of encoder convolution",
        )
        # attention
        group.add_argument(
            "--atype",
            default="location",
            type=str,
            choices=["forward_ta", "forward", "location"],
            help="Type of attention mechanism",
        )
        group.add_argument(
            "--adim",
            default=512,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aconv-chans",
            default=32,
            type=int,
            help="Number of attention convolution channels",
        )
        group.add_argument(
            "--aconv-filts",
            default=15,
            type=int,
            help="Filter size of attention convolution",
        )
        group.add_argument(
            "--cumulate-att-w",
            default=True,
            type=strtobool,
            help="Whether or not to cumulate attention weights",
        )
        # decoder
        group.add_argument(
            "--dlayers", default=2, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=1024, type=int, help="Number of decoder hidden units"
        )
        group.add_argument(
            "--prenet-layers", default=2, type=int, help="Number of prenet layers"
        )
        group.add_argument(
            "--prenet-units",
            default=256,
            type=int,
            help="Number of prenet hidden units",
        )
        group.add_argument(
            "--postnet-layers", default=5, type=int, help="Number of postnet layers"
        )
        group.add_argument(
            "--postnet-chans", default=512, type=int, help="Number of postnet channels"
        )
        group.add_argument(
            "--postnet-filts", default=5, type=int, help="Filter size of postnet"
        )
        group.add_argument(
            "--output-activation",
            default=None,
            type=str,
            nargs="?",
            help="Output activation function",
        )
        # cbhg
        group.add_argument(
            "--use-cbhg",
            default=False,
            type=strtobool,
            help="Whether to use CBHG module",
        )
        group.add_argument(
            "--cbhg-conv-bank-layers",
            default=8,
            type=int,
            help="Number of convoluional bank layers in CBHG",
        )
        group.add_argument(
            "--cbhg-conv-bank-chans",
            default=128,
            type=int,
            help="Number of convoluional bank channles in CBHG",
        )
        group.add_argument(
            "--cbhg-conv-proj-filts",
            default=3,
            type=int,
            help="Filter size of convoluional projection layer in CBHG",
        )
        group.add_argument(
            "--cbhg-conv-proj-chans",
            default=256,
            type=int,
            help="Number of convoluional projection channels in CBHG",
        )
        group.add_argument(
            "--cbhg-highway-layers",
            default=4,
            type=int,
            help="Number of highway layers in CBHG",
        )
        group.add_argument(
            "--cbhg-highway-units",
            default=128,
            type=int,
            help="Number of highway units in CBHG",
        )
        group.add_argument(
            "--cbhg-gru-units",
            default=256,
            type=int,
            help="Number of GRU units in CBHG",
        )
        # model (parameter) related
        group.add_argument(
            "--use-batch-norm",
            default=True,
            type=strtobool,
            help="Whether to use batch normalization",
        )
        group.add_argument(
            "--use-concate",
            default=True,
            type=strtobool,
            help="Whether to concatenate encoder embedding with decoder outputs",
        )
        group.add_argument(
            "--use-residual",
            default=True,
            type=strtobool,
            help="Whether to use residual connection in conv layer",
        )
        group.add_argument(
            "--dropout-rate", default=0.5, type=float, help="Dropout rate"
        )
        group.add_argument(
            "--zoneout-rate", default=0.1, type=float, help="Zoneout rate"
        )
        group.add_argument(
            "--reduction-factor",
            default=1,
            type=int,
            help="Reduction factor (for decoder)",
        )
        group.add_argument(
            "--encoder-reduction-factor",
            default=1,
            type=int,
            help="Reduction factor (for encoder)",
        )
        group.add_argument(
            "--spk-embed-dim",
            default=None,
            type=int,
            help="Number of speaker embedding dimensions",
        )
        group.add_argument(
            "--spc-dim", default=None, type=int, help="Number of spectrogram dimensions"
        )
        group.add_argument(
            "--pretrained-model", default=None, type=str, help="Pretrained model path"
        )
        # loss related
        group.add_argument(
            "--use-masking",
            default=False,
            type=strtobool,
            help="Whether to use masking in calculation of loss",
        )
        group.add_argument(
            "--bce-pos-weight",
            default=20.0,
            type=float,
            help="Positive sample weight in BCE calculation "
            "(only for use-masking=True)",
        )
        group.add_argument(
            "--use-guided-attn-loss",
            default=False,
            type=strtobool,
            help="Whether to use guided attention loss",
        )
        group.add_argument(
            "--guided-attn-loss-sigma",
            default=0.4,
            type=float,
            help="Sigma in guided attention loss",
        )
        group.add_argument(
            "--guided-attn-loss-lambda",
            default=1.0,
            type=float,
            help="Lambda in guided attention loss",
        )
        group.add_argument(
            "--src-reconstruction-loss-lambda",
            default=1.0,
            type=float,
            help="Lambda in source reconstruction loss",
        )
        group.add_argument(
            "--trg-reconstruction-loss-lambda",
            default=1.0,
            type=float,
            help="Lambda in target reconstruction loss",
        )
        return parser

    def __init__(self, idim, odim, args=None):
        """Initialize Tacotron2 module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            args (Namespace, optional):
                - spk_embed_dim (int): Dimension of the speaker embedding.
                - elayers (int): The number of encoder blstm layers.
                - eunits (int): The number of encoder blstm units.
                - econv_layers (int): The number of encoder conv layers.
                - econv_filts (int): The number of encoder conv filter size.
                - econv_chans (int): The number of encoder conv filter channels.
                - dlayers (int): The number of decoder lstm layers.
                - dunits (int): The number of decoder lstm units.
                - prenet_layers (int): The number of prenet layers.
                - prenet_units (int): The number of prenet units.
                - postnet_layers (int): The number of postnet layers.
                - postnet_filts (int): The number of postnet filter size.
                - postnet_chans (int): The number of postnet filter channels.
                - output_activation (int): The name of activation function for outputs.
                - adim (int): The number of dimension of mlp in attention.
                - aconv_chans (int): The number of attention conv filter channels.
                - aconv_filts (int): The number of attention conv filter size.
                - cumulate_att_w (bool): Whether to cumulate previous attention weight.
                - use_batch_norm (bool): Whether to use batch normalization.
                - use_concate (int):
                    Whether to concatenate encoder embedding with decoder lstm outputs.
                - dropout_rate (float): Dropout rate.
                - zoneout_rate (float): Zoneout rate.
                - reduction_factor (int): Reduction factor.
                - spk_embed_dim (int): Number of speaker embedding dimenstions.
                - spc_dim (int): Number of spectrogram embedding dimenstions
                    (only for use_cbhg=True).
                - use_cbhg (bool): Whether to use CBHG module.
                - cbhg_conv_bank_layers (int):
                    The number of convoluional banks in CBHG.
                - cbhg_conv_bank_chans (int):
                    The number of channels of convolutional bank in CBHG.
                - cbhg_proj_filts (int):
                    The number of filter size of projection layeri in CBHG.
                - cbhg_proj_chans (int):
                    The number of channels of projection layer in CBHG.
                - cbhg_highway_layers (int):
                    The number of layers of highway network in CBHG.
                - cbhg_highway_units (int):
                    The number of units of highway network in CBHG.
                - cbhg_gru_units (int): The number of units of GRU in CBHG.
                - use_masking (bool): Whether to mask padded part in loss calculation.
                - bce_pos_weight (float): Weight of positive sample of stop token
                    (only for use_masking=True).
                - use-guided-attn-loss (bool): Whether to use guided attention loss.
                - guided-attn-loss-sigma (float) Sigma in guided attention loss.
                - guided-attn-loss-lamdba (float): Lambda in guided attention loss.

        """
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # fill missing arguments
        args = fill_missing_args(args, self.add_arguments)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.adim = args.adim
        self.spk_embed_dim = args.spk_embed_dim
        self.cumulate_att_w = args.cumulate_att_w
        self.reduction_factor = args.reduction_factor
        self.encoder_reduction_factor = args.encoder_reduction_factor
        self.use_cbhg = args.use_cbhg
        self.use_guided_attn_loss = args.use_guided_attn_loss
        self.src_reconstruction_loss_lambda = args.src_reconstruction_loss_lambda
        self.trg_reconstruction_loss_lambda = args.trg_reconstruction_loss_lambda

        # define activation function for the final output
        if args.output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, args.output_activation):
            self.output_activation_fn = getattr(F, args.output_activation)
        else:
            raise ValueError(
                "there is no such an activation function. (%s)" % args.output_activation
            )

        # define network modules
        self.enc = Encoder(
            idim=idim * args.encoder_reduction_factor,
            input_layer="linear",
            elayers=args.elayers,
            eunits=args.eunits,
            econv_layers=args.econv_layers,
            econv_chans=args.econv_chans,
            econv_filts=args.econv_filts,
            use_batch_norm=args.use_batch_norm,
            use_residual=args.use_residual,
            dropout_rate=args.dropout_rate,
        )
        dec_idim = (
            args.eunits
            if args.spk_embed_dim is None
            else args.eunits + args.spk_embed_dim
        )
        if args.atype == "location":
            att = AttLoc(
                dec_idim, args.dunits, args.adim, args.aconv_chans, args.aconv_filts
            )
        elif args.atype == "forward":
            att = AttForward(
                dec_idim, args.dunits, args.adim, args.aconv_chans, args.aconv_filts
            )
            if self.cumulate_att_w:
                logging.warning(
                    "cumulation of attention weights is disabled in forward attention."
                )
                self.cumulate_att_w = False
        elif args.atype == "forward_ta":
            att = AttForwardTA(
                dec_idim,
                args.dunits,
                args.adim,
                args.aconv_chans,
                args.aconv_filts,
                odim,
            )
            if self.cumulate_att_w:
                logging.warning(
                    "cumulation of attention weights is disabled in forward attention."
                )
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
        self.dec = Decoder(
            idim=dec_idim,
            odim=odim,
            att=att,
            dlayers=args.dlayers,
            dunits=args.dunits,
            prenet_layers=args.prenet_layers,
            prenet_units=args.prenet_units,
            postnet_layers=args.postnet_layers,
            postnet_chans=args.postnet_chans,
            postnet_filts=args.postnet_filts,
            output_activation_fn=self.output_activation_fn,
            cumulate_att_w=self.cumulate_att_w,
            use_batch_norm=args.use_batch_norm,
            use_concate=args.use_concate,
            dropout_rate=args.dropout_rate,
            zoneout_rate=args.zoneout_rate,
            reduction_factor=args.reduction_factor,
        )
        self.taco2_loss = Tacotron2Loss(
            use_masking=args.use_masking, bce_pos_weight=args.bce_pos_weight
        )
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(
                sigma=args.guided_attn_loss_sigma,
                alpha=args.guided_attn_loss_lambda,
            )
        if self.use_cbhg:
            self.cbhg = CBHG(
                idim=odim,
                odim=args.spc_dim,
                conv_bank_layers=args.cbhg_conv_bank_layers,
                conv_bank_chans=args.cbhg_conv_bank_chans,
                conv_proj_filts=args.cbhg_conv_proj_filts,
                conv_proj_chans=args.cbhg_conv_proj_chans,
                highway_layers=args.cbhg_highway_layers,
                highway_units=args.cbhg_highway_units,
                gru_units=args.cbhg_gru_units,
            )
            self.cbhg_loss = CBHGLoss(use_masking=args.use_masking)
        if self.src_reconstruction_loss_lambda > 0:
            self.src_reconstructor = Encoder(
                idim=dec_idim,
                input_layer="linear",
                elayers=args.elayers,
                eunits=args.eunits,
                econv_layers=args.econv_layers,
                econv_chans=args.econv_chans,
                econv_filts=args.econv_filts,
                use_batch_norm=args.use_batch_norm,
                use_residual=args.use_residual,
                dropout_rate=args.dropout_rate,
            )
            self.src_reconstructor_linear = torch.nn.Linear(
                args.econv_chans, idim * args.encoder_reduction_factor
            )

            self.src_reconstruction_loss = CBHGLoss(use_masking=args.use_masking)
        if self.trg_reconstruction_loss_lambda > 0:
            self.trg_reconstructor = Encoder(
                idim=dec_idim,
                input_layer="linear",
                elayers=args.elayers,
                eunits=args.eunits,
                econv_layers=args.econv_layers,
                econv_chans=args.econv_chans,
                econv_filts=args.econv_filts,
                use_batch_norm=args.use_batch_norm,
                use_residual=args.use_residual,
                dropout_rate=args.dropout_rate,
            )
            self.trg_reconstructor_linear = torch.nn.Linear(
                args.econv_chans, odim * args.reduction_factor
            )
            self.trg_reconstruction_loss = CBHGLoss(use_masking=args.use_masking)

        # load pretrained model
        if args.pretrained_model is not None:
            self.load_pretrained_model(args.pretrained_model)

    def forward(
        self, xs, ilens, ys, labels, olens, spembs=None, spcs=None, *args, **kwargs
    ):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded acoustic features (B, Tmax, idim).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional):
                Batch of speaker embedding vectors (B, spk_embed_dim).
            spcs (Tensor, optional):
                Batch of groundtruth spectrograms (B, Lmax, spc_dim).

        Returns:
            Tensor: Loss value.

        """
        # remove unnecessary padded part (for multi-gpus)
        max_in = max(ilens)
        max_out = max(olens)
        if max_in != xs.shape[1]:
            xs = xs[:, :max_in]
        if max_out != ys.shape[1]:
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]

        # thin out input frames for reduction factor
        # (B, Lmax, idim) ->  (B, Lmax // r, idim * r)
        if self.encoder_reduction_factor > 1:
            B, Lmax, idim = xs.shape
            if Lmax % self.encoder_reduction_factor != 0:
                xs = xs[:, : -(Lmax % self.encoder_reduction_factor), :]
            xs_ds = xs.contiguous().view(
                B,
                int(Lmax / self.encoder_reduction_factor),
                idim * self.encoder_reduction_factor,
            )
            ilens_ds = ilens.new(
                [ilen // self.encoder_reduction_factor for ilen in ilens]
            )
        else:
            xs_ds, ilens_ds = xs, ilens

        # calculate tacotron2 outputs
        hs, hlens = self.enc(xs_ds, ilens_ds)
        if self.spk_embed_dim is not None:
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        after_outs, before_outs, logits, att_ws = self.dec(hs, hlens, ys)

        # calculate src reconstruction
        if self.src_reconstruction_loss_lambda > 0:
            B, _in_length, _adim = hs.shape
            xt, xtlens = self.src_reconstructor(hs, hlens)
            xt = self.src_reconstructor_linear(xt)
            if self.encoder_reduction_factor > 1:
                xt = xt.view(B, -1, self.idim)

        # calculate trg reconstruction
        if self.trg_reconstruction_loss_lambda > 0:
            olens_trg_cp = olens.new(
                sorted([olen // self.reduction_factor for olen in olens], reverse=True)
            )
            B, _in_length, _adim = hs.shape
            _, _out_length, _ = att_ws.shape
            # att_R should be [B, out_length / r_d, adim]
            att_R = torch.sum(
                hs.view(B, 1, _in_length, _adim)
                * att_ws.view(B, _out_length, _in_length, 1),
                dim=2,
            )
            yt, ytlens = self.trg_reconstructor(
                att_R, olens_trg_cp
            )  # is using olens correct?
            yt = self.trg_reconstructor_linear(yt)
            if self.reduction_factor > 1:
                yt = yt.view(
                    B, -1, self.odim
                )  # now att_R should be [B, out_length, adim]

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            assert olens.ge(
                self.reduction_factor
            ).all(), "Output length must be greater than or equal to reduction factor."
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_out = max(olens)
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]
            labels = torch.scatter(
                labels, 1, (olens - 1).unsqueeze(1), 1.0
            )  # see #3388
        if self.encoder_reduction_factor > 1:
            ilens = ilens.new(
                [ilen - ilen % self.encoder_reduction_factor for ilen in ilens]
            )
            max_in = max(ilens)
            xs = xs[:, :max_in]

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(
            after_outs, before_outs, logits, ys, labels, olens
        )
        loss = l1_loss + mse_loss + bce_loss
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"mse_loss": mse_loss.item()},
            {"bce_loss": bce_loss.item()},
        ]

        # calculate context_preservation loss
        if self.src_reconstruction_loss_lambda > 0:
            src_recon_l1_loss, src_recon_mse_loss = self.src_reconstruction_loss(
                xt, xs, ilens
            )
            loss = loss + src_recon_l1_loss
            report_keys += [
                {"src_recon_l1_loss": src_recon_l1_loss.item()},
                {"src_recon_mse_loss": src_recon_mse_loss.item()},
            ]
        if self.trg_reconstruction_loss_lambda > 0:
            trg_recon_l1_loss, trg_recon_mse_loss = self.trg_reconstruction_loss(
                yt, ys, olens
            )
            loss = loss + trg_recon_l1_loss
            report_keys += [
                {"trg_recon_l1_loss": trg_recon_l1_loss.item()},
                {"trg_recon_mse_loss": trg_recon_mse_loss.item()},
            ]

        # calculate attention loss
        if self.use_guided_attn_loss:
            # NOTE(kan-bayashi): length of output for auto-regressive input
            #   will be changed when r > 1
            if self.encoder_reduction_factor > 1:
                ilens_in = ilens.new(
                    [ilen // self.encoder_reduction_factor for ilen in ilens]
                )
            else:
                ilens_in = ilens
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            attn_loss = self.attn_loss(att_ws, ilens_in, olens_in)
            loss = loss + attn_loss
            report_keys += [
                {"attn_loss": attn_loss.item()},
            ]

        # calculate cbhg loss
        if self.use_cbhg:
            # remove unnecessary padded part (for multi-gpus)
            if max_out != spcs.shape[1]:
                spcs = spcs[:, :max_out]

            # calculate cbhg outputs & loss and report them
            cbhg_outs, _ = self.cbhg(after_outs, olens)
            cbhg_l1_loss, cbhg_mse_loss = self.cbhg_loss(cbhg_outs, spcs, olens)
            loss = loss + cbhg_l1_loss + cbhg_mse_loss
            report_keys += [
                {"cbhg_l1_loss": cbhg_l1_loss.item()},
                {"cbhg_mse_loss": cbhg_mse_loss.item()},
            ]

        report_keys += [{"loss": loss.item()}]
        self.reporter.report(report_keys)

        return loss

    def inference(self, x, inference_args, spemb=None, *args, **kwargs):
        """Generate the sequence of features given the sequences of characters.

        Args:
            x (Tensor): Input sequence of acoustic features (T, idim).
            inference_args (Namespace):
                - threshold (float): Threshold in inference.
                - minlenratio (float): Minimum length ratio in inference.
                - maxlenratio (float): Maximum length ratio in inference.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).

        """
        # get options
        threshold = inference_args.threshold
        minlenratio = inference_args.minlenratio
        maxlenratio = inference_args.maxlenratio

        # thin out input frames for reduction factor
        # (B, Lmax, idim) ->  (B, Lmax // r, idim * r)
        if self.encoder_reduction_factor > 1:
            Lmax, idim = x.shape
            if Lmax % self.encoder_reduction_factor != 0:
                x = x[: -(Lmax % self.encoder_reduction_factor), :]
            x_ds = x.contiguous().view(
                int(Lmax / self.encoder_reduction_factor),
                idim * self.encoder_reduction_factor,
            )
        else:
            x_ds = x

        # inference
        h = self.enc.inference(x_ds)
        if self.spk_embed_dim is not None:
            spemb = F.normalize(spemb, dim=0).unsqueeze(0).expand(h.size(0), -1)
            h = torch.cat([h, spemb], dim=-1)
        outs, probs, att_ws = self.dec.inference(h, threshold, minlenratio, maxlenratio)

        if self.use_cbhg:
            cbhg_outs = self.cbhg.inference(outs)
            return cbhg_outs, probs, att_ws
        else:
            return outs, probs, att_ws

    def calculate_all_attentions(self, xs, ilens, ys, spembs=None, *args, **kwargs):
        """Calculate all of the attention weights.

        Args:
            xs (Tensor): Batch of padded acoustic features (B, Tmax, idim).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional):
                Batch of speaker embedding vectors (B, spk_embed_dim).

        Returns:
            numpy.ndarray: Batch of attention weights (B, Lmax, Tmax).

        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))

        self.eval()
        with torch.no_grad():
            # thin out input frames for reduction factor
            # (B, Lmax, idim) ->  (B, Lmax // r, idim * r)
            if self.encoder_reduction_factor > 1:
                B, Lmax, idim = xs.shape
                if Lmax % self.encoder_reduction_factor != 0:
                    xs = xs[:, : -(Lmax % self.encoder_reduction_factor), :]
                xs_ds = xs.contiguous().view(
                    B,
                    int(Lmax / self.encoder_reduction_factor),
                    idim * self.encoder_reduction_factor,
                )
                ilens_ds = [ilen // self.encoder_reduction_factor for ilen in ilens]
            else:
                xs_ds, ilens_ds = xs, ilens

            hs, hlens = self.enc(xs_ds, ilens_ds)
            if self.spk_embed_dim is not None:
                spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
                hs = torch.cat([hs, spembs], dim=-1)
            att_ws = self.dec.calculate_all_attentions(hs, hlens, ys)
        self.train()

        return att_ws.cpu().numpy()

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training.

        keys should match what `chainer.reporter` reports.
        If you add the key `loss`, the reporter will report `main/loss`
            and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss`
            and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ["loss", "l1_loss", "mse_loss", "bce_loss"]
        if self.use_guided_attn_loss:
            plot_keys += ["attn_loss"]
        if self.use_cbhg:
            plot_keys += ["cbhg_l1_loss", "cbhg_mse_loss"]
        if self.src_reconstruction_loss_lambda > 0:
            plot_keys += ["src_recon_l1_loss", "src_recon_mse_loss"]
        if self.trg_reconstruction_loss_lambda > 0:
            plot_keys += ["trg_recon_l1_loss", "trg_recon_mse_loss"]
        return plot_keys

    def _sort_by_length(self, xs, ilens):
        sort_ilens, sort_idx = ilens.sort(0, descending=True)
        return xs[sort_idx], ilens[sort_idx], sort_idx

    def _revert_sort_by_length(self, xs, ilens, sort_idx):
        _, revert_idx = sort_idx.sort(0)
        return xs[revert_idx], ilens[revert_idx]
