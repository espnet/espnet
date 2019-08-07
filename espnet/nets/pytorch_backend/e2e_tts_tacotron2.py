#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

from distutils.util import strtobool

import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.rnn.attentions import AttForward
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.rnn.attentions import AttLoc
from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHG
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.get_attribute import get_attribute


DEFAULTS = {
    "embed_dim": 512,
    "elayers": 1,
    "eunits": 512,
    "econv_layers": 3,
    "econv_filts": 5,
    "econv_chans": 512,
    "atype": "location",
    "adim": 128,
    "aconv_filts": 15,
    "aconv_chans": 32,
    "dlayers": 2,
    "dunits": 1024,
    "prenet_layers": 2,
    "prenet_units": 256,
    "postnet_layers": 5,
    "postnet_filts": 5,
    "postnet_chans": 512,
    "dropout_rate": 0.5,
    "zoneout_rate": 0.1,
    "reduction_factor": 1,
    "spk_embed_dim": None,
    "output_activation": None,
    "cumulate_att_w": True,
    "use_batch_norm": True,
    "use_concate": True,
    "use_residual": False,
    "use_masking": True,
    "bce_pos_weight": 20.0,
    "use_cbhg": False,
    "spc_dim": None,
    "cbhg_conv_bank_layers": 8,
    "cbhg_conv_bank_chans": 128,
    "cbhg_conv_proj_filts": 3,
    "cbhg_conv_proj_chans": 256,
    "cbhg_highway_layers": 4,
    "cbhg_highway_units": 128,
    "cbhg_gru_units": 256,
    "use_guided_attn_loss": False,
    "guided_attn_loss_sigma": 0.4,
}


class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.

    This module calculates the guided attention loss described in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_, which forces the attention to be diagonal.

    Args:
        sigma (float, optional): Standard deviation to control how close attention to a diagonal.
        reset_always (bool, optional): Whether to always reset masks.

    .. _`Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969

    """

    def __init__(self, sigma=0.4, reset_always=True):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = torch.tensor(sigma)
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.

        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).

        Returns:
            Tensor: Guided attention loss value.

        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(att_ws.device)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma)
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.

        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])

        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        """Make masks indicating non-padded part.

        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)

        """
        in_masks = make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = make_non_pad_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)


class CBHGLoss(torch.nn.Module):
    """Loss function module for CBHG.

    Args:
        use_masking (bool): Whether to mask padded part in loss calculation.

    """

    def __init__(self, use_masking=True):
        super(CBHGLoss, self).__init__()
        self.use_masking = use_masking

    def forward(self, cbhg_outs, spcs, olens):
        """Calculate forward propagation.

        Args:
            cbhg_outs (Tensor): Batch of CBHG outputs (B, Lmax, spc_dim).
            spcs (Tensor): Batch of groundtruth of spectrogram (B, Lmax, spc_dim).
            olens (LongTensor): Batch of the lengths of each sequence (B,).

        Returns:
            Tensor: L1 loss value
            Tensor: Mean square error loss value.

        """
        # perform masking for padded values
        if self.use_masking:
            mask = make_non_pad_mask(olens).unsqueeze(-1).to(spcs.device)
            spcs = spcs.masked_select(mask)
            cbhg_outs = cbhg_outs.masked_select(mask)

        # calculate loss
        cbhg_l1_loss = F.l1_loss(cbhg_outs, spcs)
        cbhg_mse_loss = F.mse_loss(cbhg_outs, spcs)

        return cbhg_l1_loss, cbhg_mse_loss


class Tacotron2Loss(torch.nn.Module):
    """Loss function module for Tacotron2.

    Args:
        use_masking (bool): Whether to mask padded part in loss calculation.
        bce_pos_weight (float): Weight of positive sample of stop token.

    """

    def __init__(self, use_masking=True, bce_pos_weight=20.0):
        super(Tacotron2Loss, self).__init__()
        self.use_masking = use_masking
        self.bce_pos_weight = bce_pos_weight

    def forward(self, after_outs, before_outs, logits, ys, labels, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.

        """
        # perform masking for padded values
        if self.use_masking:
            mask = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(mask)
            after_outs = after_outs.masked_select(mask)
            before_outs = before_outs.masked_select(mask)
            labels = labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])

        # calculate loss
        l1_loss = F.l1_loss(after_outs, ys) + F.l1_loss(before_outs, ys)
        mse_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(self.bce_pos_weight, device=ys.device))

        return l1_loss, mse_loss, bce_loss


class Tacotron2(TTSInterface, torch.nn.Module):
    """Tacotron2 module for end-to-end text-to-speech (E2E-TTS).

    This is a module of Spectrogram prediction network in Tacotron2 described in `Natural TTS Synthesis
    by Conditioning WaveNet on Mel Spectrogram Predictions`_, which converts the sequence of characters
    into the sequence of Mel-filterbanks.

    Args:
        idim (int): Dimension of the inputs.
        odim (int): Dimension of the outputs.
        args (Namespace):
            - spk_embed_dim (int): Dimension of the speaker embedding.
            - embed_dim (int): Dimension of character embedding.
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
            - use_concate (int): Whether to concatenate encoder embedding with decoder lstm outputs.
            - dropout_rate (float): Dropout rate.
            - zoneout_rate (float): Zoneout rate.
            - reduction_factor (int): Reduction factor.
            - use_cbhg (bool): Whether to use CBHG module.
            - cbhg_conv_bank_layers (int): The number of convoluional banks in CBHG.
            - cbhg_conv_bank_chans (int): The number of channels of convolutional bank in CBHG.
            - cbhg_proj_filts (int): The number of filter size of projection layeri in CBHG.
            - cbhg_proj_chans (int): The number of channels of projection layer in CBHG.
            - cbhg_highway_layers (int): The number of layers of highway network in CBHG.
            - cbhg_highway_units (int): The number of units of highway network in CBHG.
            - cbhg_gru_units (int): The number of units of GRU in CBHG.
            - use_masking (bool): Whether to mask padded part in loss calculation.
            - bce_pos_weight (float): Weight of positive sample of stop token (only for use_masking=True).
            - use-guided-attn-loss (bool): Whether to use guided attention loss.
            - guided-attn-loss-sigma (float) Sigma in guided attention loss.
            - guided-attn-loss-lamdba (float): Lambda in guided attention loss.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    @staticmethod
    def add_arguments(parser):
        """Add model-specific arguments to the parser."""
        # encoder
        parser.add_argument('--embed-dim', default=DEFAULTS["embed_dim"], type=int,
                            help='Number of dimension of embedding')
        parser.add_argument('--elayers', default=DEFAULTS["elayers"], type=int,
                            help='Number of encoder layers')
        parser.add_argument('--eunits', '-u', default=DEFAULTS["eunits"], type=int,
                            help='Number of encoder hidden units')
        parser.add_argument('--econv-layers', default=DEFAULTS["econv_layers"], type=int,
                            help='Number of encoder convolution layers')
        parser.add_argument('--econv-chans', default=DEFAULTS["econv_chans"], type=int,
                            help='Number of encoder convolution channels')
        parser.add_argument('--econv-filts', default=DEFAULTS["econv_filts"], type=int,
                            help='Filter size of encoder convolution')
        # attention
        parser.add_argument('--atype', default=DEFAULTS["atype"], type=str,
                            choices=["forward_ta", "forward", "location"],
                            help='Type of attention mechanism')
        parser.add_argument('--adim', default=DEFAULTS["adim"], type=int,
                            help='Number of attention transformation dimensions')
        parser.add_argument('--aconv-chans', default=DEFAULTS["aconv_chans"], type=int,
                            help='Number of attention convolution channels')
        parser.add_argument('--aconv-filts', default=DEFAULTS["aconv_filts"], type=int,
                            help='Filter size of attention convolution')
        parser.add_argument('--cumulate-att-w', default=DEFAULTS["cumulate_att_w"], type=strtobool,
                            help="Whether or not to cumulate attention weights")
        # decoder
        parser.add_argument('--dlayers', default=DEFAULTS["dlayers"], type=int,
                            help='Number of decoder layers')
        parser.add_argument('--dunits', default=DEFAULTS["dunits"], type=int,
                            help='Number of decoder hidden units')
        parser.add_argument('--prenet-layers', default=DEFAULTS["prenet_layers"], type=int,
                            help='Number of prenet layers')
        parser.add_argument('--prenet-units', default=DEFAULTS["prenet_units"], type=int,
                            help='Number of prenet hidden units')
        parser.add_argument('--postnet-layers', default=DEFAULTS["postnet_layers"], type=int,
                            help='Number of postnet layers')
        parser.add_argument('--postnet-chans', default=DEFAULTS["postnet_chans"], type=int,
                            help='Number of postnet channels')
        parser.add_argument('--postnet-filts', default=DEFAULTS["postnet_filts"], type=int,
                            help='Filter size of postnet')
        parser.add_argument('--output-activation', default=DEFAULTS["output_activation"], nargs='?',
                            help='Output activation function')
        # cbhg
        parser.add_argument('--use-cbhg', default=DEFAULTS["use_cbhg"], type=strtobool,
                            help='Whether to use CBHG module')
        parser.add_argument('--cbhg-conv-bank-layers', default=DEFAULTS["cbhg_conv_bank_layers"], type=int,
                            help='Number of convoluional bank layers in CBHG')
        parser.add_argument('--cbhg-conv-bank-chans', default=DEFAULTS["cbhg_conv_bank_chans"], type=int,
                            help='Number of convoluional bank channles in CBHG')
        parser.add_argument('--cbhg-conv-proj-filts', default=DEFAULTS["cbhg_conv_proj_filts"], type=int,
                            help='Filter size of convoluional projection layer in CBHG')
        parser.add_argument('--cbhg-conv-proj-chans', default=DEFAULTS["cbhg_conv_proj_chans"], type=int,
                            help='Number of convoluional projection channels in CBHG')
        parser.add_argument('--cbhg-highway-layers', default=DEFAULTS["cbhg_highway_layers"], type=int,
                            help='Number of highway layers in CBHG')
        parser.add_argument('--cbhg-highway-units', default=DEFAULTS["cbhg_highway_units"], type=int,
                            help='Number of highway units in CBHG')
        parser.add_argument('--cbhg-gru-units', default=DEFAULTS["cbhg_gru_units"], type=int,
                            help='Number of GRU units in CBHG')
        # model (parameter) related
        parser.add_argument('--use-batch-norm', default=DEFAULTS["use_batch_norm"], type=strtobool,
                            help='Whether to use batch normalization')
        parser.add_argument('--use-concate', default=DEFAULTS["use_concate"], type=strtobool,
                            help='Whether to concatenate encoder embedding with decoder outputs')
        parser.add_argument('--use-residual', default=DEFAULTS["use_residual"], type=strtobool,
                            help='Whether to use residual connection in conv layer')
        parser.add_argument('--dropout-rate', default=DEFAULTS["dropout_rate"], type=float,
                            help='Dropout rate')
        parser.add_argument('--zoneout-rate', default=DEFAULTS["zoneout_rate"], type=float,
                            help='Zoneout rate')
        parser.add_argument('--reduction-factor', default=DEFAULTS["reduction_factor"], type=int,
                            help='Reduction factor')
        # loss related
        parser.add_argument('--use-masking', default=DEFAULTS["use_masking"], type=strtobool,
                            help='Whether to use masking in calculation of loss')
        parser.add_argument('--bce-pos-weight', default=DEFAULTS["bce_pos_weight"], type=float,
                            help='Positive sample weight in BCE calculation (only for use-masking=True)')
        parser.add_argument("--use-guided-attn-loss", default=DEFAULTS["use_guided_attn_loss"], type=strtobool,
                            help="Whether to use guided attention loss")
        parser.add_argument("--guided-attn-loss-sigma", default=DEFAULTS["guided_attn_loss_sigma"], type=float,
                            help="Sigma in guided attention loss")
        return

    def __init__(self, idim, odim, args):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # get hyperparameters
        embed_dim = get_attribute(args, "embed_dim", DEFAULTS["embed_dim"])
        elayers = get_attribute(args, "elayers", DEFAULTS["elayers"])
        eunits = get_attribute(args, "eunits", DEFAULTS["eunits"])
        econv_layers = get_attribute(args, "econv_layers", DEFAULTS["econv_layers"])
        econv_filts = get_attribute(args, "econv_filts", DEFAULTS["econv_filts"])
        econv_chans = get_attribute(args, "econv_chans", DEFAULTS["econv_chans"])
        atype = get_attribute(args, "atype", DEFAULTS["atype"])
        adim = get_attribute(args, "adim", DEFAULTS["adim"])
        aconv_filts = get_attribute(args, "aconv_filts", DEFAULTS["aconv_filts"])
        aconv_chans = get_attribute(args, "aconv_chans", DEFAULTS["aconv_chans"])
        dlayers = get_attribute(args, "dlayers", DEFAULTS["dlayers"])
        dunits = get_attribute(args, "dunits", DEFAULTS["dunits"])
        prenet_layers = get_attribute(args, "prenet_layers", DEFAULTS["prenet_layers"])
        prenet_units = get_attribute(args, "prenet_units", DEFAULTS["prenet_units"])
        postnet_layers = get_attribute(args, "postnet_layers", DEFAULTS["postnet_layers"])
        postnet_filts = get_attribute(args, "postnet_filts", DEFAULTS["postnet_filts"])
        postnet_chans = get_attribute(args, "postnet_chans", DEFAULTS["postnet_chans"])
        dropout_rate = get_attribute(args, "dropout_rate", DEFAULTS["dropout_rate"])
        zoneout_rate = get_attribute(args, "zoneout_rate", DEFAULTS["zoneout_rate"])
        reduction_factor = get_attribute(args, "reduction_factor", DEFAULTS["reduction_factor"])
        spk_embed_dim = get_attribute(args, "spk_embed_dim", DEFAULTS["spk_embed_dim"])
        output_activation = get_attribute(args, "output_activation", DEFAULTS["output_activation"])
        cumulate_att_w = get_attribute(args, "cumulate_att_w", DEFAULTS["cumulate_att_w"])
        use_batch_norm = get_attribute(args, "use_batch_norm", DEFAULTS["use_batch_norm"])
        use_concate = get_attribute(args, "use_concate", DEFAULTS["use_concate"])
        use_residual = get_attribute(args, "use_residual", DEFAULTS["use_residual"])
        use_masking = get_attribute(args, "use_masking", DEFAULTS["use_masking"])
        bce_pos_weight = get_attribute(args, "bce_pos_weight", DEFAULTS["bce_pos_weight"])
        use_cbhg = get_attribute(args, "use_cbhg", DEFAULTS["use_cbhg"])
        if use_cbhg:
            spc_dim = get_attribute(args, DEFAULTS["spc_dim"])
            cbhg_conv_bank_layers = get_attribute(args, "cbhg_conv_bank_layers", DEFAULTS["cbhg_conv_bank_layers"])
            cbhg_conv_bank_chans = get_attribute(args, "cbhg_conv_bank_chans", DEFAULTS["cbhg_conv_bank_chans"])
            cbhg_conv_proj_filts = get_attribute(args, "cbhg_conv_proj_filts", DEFAULTS["cbhg_conv_proj_filts"])
            cbhg_conv_proj_chans = get_attribute(args, "cbhg_conv_proj_chans", DEFAULTS["cbhg_conv_proj_chans"])
            cbhg_highway_layers = get_attribute(args, "cbhg_highway_layers", DEFAULTS["cbhg_highway_layers"])
            cbhg_highway_units = get_attribute(args, "cbhg_highway_units", DEFAULTS["cbhg_highway_units"])
            cbhg_gru_units = get_attribute(args, "cbhg_gru_units", DEFAULTS["cbhg_gru_units"])
        use_guided_attn_loss = get_attribute(args, "use_guided_attn_loss", DEFAULTS["use_guided_attn_loss"])
        if use_guided_attn_loss:
            guided_attn_loss_sigma = get_attribute(args, "guided_attn_loss_sigma", DEFAULTS["guided_attn_loss_sigma"])

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.spk_embed_dim = spk_embed_dim
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor
        self.use_cbhg = use_cbhg
        self.use_guided_attn_loss = use_guided_attn_loss

        # define activation function for the final output
        if output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, output_activation):
            self.output_activation_fn = getattr(F, output_activation)
        else:
            raise ValueError('there is no such an activation function. (%s)' % output_activation)

        # set padding idx
        padding_idx = 0

        # define network modules
        self.enc = Encoder(idim=idim,
                           embed_dim=embed_dim,
                           elayers=elayers,
                           eunits=eunits,
                           econv_layers=econv_layers,
                           econv_chans=econv_chans,
                           econv_filts=econv_filts,
                           use_batch_norm=use_batch_norm,
                           use_residual=use_residual,
                           dropout_rate=dropout_rate,
                           padding_idx=padding_idx)
        dec_idim = eunits if spk_embed_dim is None else eunits + spk_embed_dim
        if atype == "location":
            att = AttLoc(dec_idim,
                         dunits,
                         adim,
                         aconv_chans,
                         aconv_filts)
        elif atype == "forward":
            att = AttForward(dec_idim,
                             dunits,
                             adim,
                             aconv_chans,
                             aconv_filts)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled in forward attention.")
                self.cumulate_att_w = False
        elif atype == "forward_ta":
            att = AttForwardTA(dec_idim,
                               dunits,
                               adim,
                               aconv_chans,
                               aconv_filts,
                               odim)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled in forward attention.")
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
        self.dec = Decoder(idim=dec_idim,
                           odim=odim,
                           att=att,
                           dlayers=dlayers,
                           dunits=dunits,
                           prenet_layers=prenet_layers,
                           prenet_units=prenet_units,
                           postnet_layers=postnet_layers,
                           postnet_chans=postnet_chans,
                           postnet_filts=postnet_filts,
                           output_activation_fn=self.output_activation_fn,
                           cumulate_att_w=self.cumulate_att_w,
                           use_batch_norm=use_batch_norm,
                           use_concate=use_concate,
                           dropout_rate=dropout_rate,
                           zoneout_rate=zoneout_rate,
                           reduction_factor=reduction_factor)
        self.taco2_loss = Tacotron2Loss(use_masking=use_masking,
                                        bce_pos_weight=bce_pos_weight)
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(sigma=guided_attn_loss_sigma)
        if self.use_cbhg:
            self.cbhg = CBHG(idim=odim,
                             odim=spc_dim,
                             conv_bank_layers=cbhg_conv_bank_layers,
                             conv_bank_chans=cbhg_conv_bank_chans,
                             conv_proj_filts=cbhg_conv_proj_filts,
                             conv_proj_chans=cbhg_conv_proj_chans,
                             highway_layers=cbhg_highway_layers,
                             highway_units=cbhg_highway_units,
                             gru_units=cbhg_gru_units)
            self.cbhg_loss = CBHGLoss(use_masking=use_masking)

    def forward(self, xs, ilens, ys, labels, olens, spembs=None, spcs=None, *args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).
            spcs (Tensor, optional): Batch of groundtruth spectrograms (B, Lmax, spc_dim).

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

        # calculate tacotron2 outputs
        hs, hlens = self.enc(xs, ilens)
        if self.spk_embed_dim is not None:
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        after_outs, before_outs, logits, att_ws = self.dec(hs, hlens, ys)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_out = max(olens)
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]
            labels[:, -1] = 1.0  # make sure at least one frame has 1

        # caluculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(
            after_outs, before_outs, logits, ys, labels, olens)
        loss = l1_loss + mse_loss + bce_loss
        report_keys = [
            {'l1_loss': l1_loss.item()},
            {'mse_loss': mse_loss.item()},
            {'bce_loss': bce_loss.item()},
        ]

        # caluculate attention loss
        if self.use_guided_attn_loss:
            attn_loss = self.attn_loss(att_ws, ilens, olens)
            loss = loss + attn_loss
            report_keys += [
                {'attn_loss': attn_loss.item()},
            ]

        # caluculate cbhg loss
        if self.use_cbhg:
            # remove unnecessary padded part (for multi-gpus)
            if max_out != spcs.shape[1]:
                spcs = spcs[:, :max_out]

            # caluculate cbhg outputs & loss and report them
            cbhg_outs, _ = self.cbhg(after_outs, olens)
            cbhg_l1_loss, cbhg_mse_loss = self.cbhg_loss(cbhg_outs, spcs, olens)
            loss = loss + cbhg_l1_loss + cbhg_mse_loss
            report_keys += [
                {'cbhg_l1_loss': cbhg_l1_loss.item()},
                {'cbhg_mse_loss': cbhg_mse_loss.item()},
            ]

        report_keys += [{'loss': loss.item()}]
        self.reporter.report(report_keys)

        return loss

    def inference(self, x, inference_args, spemb=None, *args, **kwargs):
        """Generate the sequence of features given the sequences of characters.

        Args:
            x (Tensor): Input sequence of characters (T,).
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
        threshold = get_attribute(inference_args, "threshold", 0.5)
        minlenratio = get_attribute(inference_args, "minlenratio", 0.0)
        maxlenratio = get_attribute(inference_args, "maxlenratio", 10.0)

        # inference
        h = self.enc.inference(x)
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
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).

        Returns:
            numpy.ndarray: Batch of attention weights (B, Lmax, Tmax).

        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))

        self.eval()
        with torch.no_grad():
            hs, hlens = self.enc(xs, ilens)
            if self.spk_embed_dim is not None:
                spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
                hs = torch.cat([hs, spembs], dim=-1)
            att_ws = self.dec.calculate_all_attentions(hs, hlens, ys)
        self.train()

        return att_ws.cpu().numpy()

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training. keys should match what `chainer.reporter` reports.

        If you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ['loss', 'l1_loss', 'mse_loss', 'bce_loss']
        if self.use_guided_attn_loss:
            plot_keys += ['attn_loss']
        if self.use_cbhg:
            plot_keys += ['cbhg_l1_loss', 'cbhg_mse_loss']
        return plot_keys
