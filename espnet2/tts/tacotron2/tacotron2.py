# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules for ESPnet2."""

import logging
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.gst.style_encoder import StyleEncoder
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import (
    GuidedAttentionLoss,
    Tacotron2Loss,
)
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.attentions import AttForward, AttForwardTA, AttLoc
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder


class Tacotron2(AbsTTS):
    """Tacotron2 module for end-to-end text-to-speech.

    This is a module of Spectrogram prediction network in Tacotron2 described
    in `Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_,
    which converts the sequence of characters into the sequence of Mel-filterbanks.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        embed_dim: int = 512,
        elayers: int = 1,
        eunits: int = 512,
        econv_layers: int = 3,
        econv_chans: int = 512,
        econv_filts: int = 5,
        atype: str = "location",
        adim: int = 512,
        aconv_chans: int = 32,
        aconv_filts: int = 15,
        cumulate_att_w: bool = True,
        dlayers: int = 2,
        dunits: int = 1024,
        prenet_layers: int = 2,
        prenet_units: int = 256,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        output_activation: str = None,
        use_batch_norm: bool = True,
        use_concate: bool = True,
        use_residual: bool = False,
        reduction_factor: int = 1,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "concat",
        use_gst: bool = False,
        gst_tokens: int = 10,
        gst_heads: int = 4,
        gst_conv_layers: int = 6,
        gst_conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        gst_conv_kernel_size: int = 3,
        gst_conv_stride: int = 2,
        gst_gru_layers: int = 1,
        gst_gru_units: int = 128,
        # training related
        dropout_rate: float = 0.5,
        zoneout_rate: float = 0.1,
        use_masking: bool = True,
        use_weighted_masking: bool = False,
        bce_pos_weight: float = 5.0,
        loss_type: str = "L1+L2",
        use_guided_attn_loss: bool = True,
        guided_attn_loss_sigma: float = 0.4,
        guided_attn_loss_lambda: float = 1.0,
    ):
        """Initialize Tacotron2 module.

        Args:
            idim (int): Dimension of the inputs.
            odim: (int) Dimension of the outputs.
            embed_dim (int): Dimension of the token embedding.
            elayers (int): Number of encoder blstm layers.
            eunits (int): Number of encoder blstm units.
            econv_layers (int): Number of encoder conv layers.
            econv_filts (int): Number of encoder conv filter size.
            econv_chans (int): Number of encoder conv filter channels.
            dlayers (int): Number of decoder lstm layers.
            dunits (int): Number of decoder lstm units.
            prenet_layers (int): Number of prenet layers.
            prenet_units (int): Number of prenet units.
            postnet_layers (int): Number of postnet layers.
            postnet_filts (int): Number of postnet filter size.
            postnet_chans (int): Number of postnet filter channels.
            output_activation (str): Name of activation function for outputs.
            adim (int): Number of dimension of mlp in attention.
            aconv_chans (int): Number of attention conv filter channels.
            aconv_filts (int): Number of attention conv filter size.
            cumulate_att_w (bool): Whether to cumulate previous attention weight.
            use_batch_norm (bool): Whether to use batch normalization.
            use_concate (bool): Whether to concat enc outputs w/ dec lstm outputs.
            reduction_factor (int): Reduction factor.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type (str): How to integrate speaker embedding.
            use_gst (str): Whether to use global style token.
            gst_tokens (int): Number of GST embeddings.
            gst_heads (int): Number of heads in GST multihead attention.
            gst_conv_layers (int): Number of conv layers in GST.
            gst_conv_chans_list: (Sequence[int]): List of the number of channels of conv
                layers in GST.
            gst_conv_kernel_size (int): Kernel size of conv layers in GST.
            gst_conv_stride (int): Stride size of conv layers in GST.
            gst_gru_layers (int): Number of GRU layers in GST.
            gst_gru_units (int): Number of GRU units in GST.
            dropout_rate (float): Dropout rate.
            zoneout_rate (float): Zoneout rate.
            use_masking (bool): Whether to mask padded part in loss calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in
                loss calculation.
            bce_pos_weight (float): Weight of positive sample of stop token
                (only for use_masking=True).
            loss_type (str): Loss function type ("L1", "L2", or "L1+L2").
            use_guided_attn_loss (bool): Whether to use guided attention loss.
            guided_attn_loss_sigma (float): Sigma in guided attention loss.
            guided_attn_loss_lambda (float): Lambda in guided attention loss.

        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor
        self.use_gst = use_gst
        self.use_guided_attn_loss = use_guided_attn_loss
        self.loss_type = loss_type

        # define activation function for the final output
        if output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, output_activation):
            self.output_activation_fn = getattr(F, output_activation)
        else:
            raise ValueError(
                f"there is no such an activation function. " f"({output_activation})"
            )

        # set padding idx
        padding_idx = 0
        self.padding_idx = padding_idx

        # define network modules
        self.enc = Encoder(
            idim=idim,
            embed_dim=embed_dim,
            elayers=elayers,
            eunits=eunits,
            econv_layers=econv_layers,
            econv_chans=econv_chans,
            econv_filts=econv_filts,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            dropout_rate=dropout_rate,
            padding_idx=padding_idx,
        )

        if self.use_gst:
            self.gst = StyleEncoder(
                idim=odim,  # the input is mel-spectrogram
                gst_tokens=gst_tokens,
                gst_token_dim=eunits,
                gst_heads=gst_heads,
                conv_layers=gst_conv_layers,
                conv_chans_list=gst_conv_chans_list,
                conv_kernel_size=gst_conv_kernel_size,
                conv_stride=gst_conv_stride,
                gru_layers=gst_gru_layers,
                gru_units=gst_gru_units,
            )

        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, eunits)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, eunits)

        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is None:
            dec_idim = eunits
        elif self.spk_embed_integration_type == "concat":
            dec_idim = eunits + spk_embed_dim
        elif self.spk_embed_integration_type == "add":
            dec_idim = eunits
            self.projection = torch.nn.Linear(self.spk_embed_dim, eunits)
        else:
            raise ValueError(f"{spk_embed_integration_type} is not supported.")

        if atype == "location":
            att = AttLoc(dec_idim, dunits, adim, aconv_chans, aconv_filts)
        elif atype == "forward":
            att = AttForward(dec_idim, dunits, adim, aconv_chans, aconv_filts)
            if self.cumulate_att_w:
                logging.warning(
                    "cumulation of attention weights is disabled "
                    "in forward attention."
                )
                self.cumulate_att_w = False
        elif atype == "forward_ta":
            att = AttForwardTA(dec_idim, dunits, adim, aconv_chans, aconv_filts, odim)
            if self.cumulate_att_w:
                logging.warning(
                    "cumulation of attention weights is disabled "
                    "in forward attention."
                )
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
        self.dec = Decoder(
            idim=dec_idim,
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
            reduction_factor=reduction_factor,
        )
        self.taco2_loss = Tacotron2Loss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(
                sigma=guided_attn_loss_sigma, alpha=guided_attn_loss_lambda,
            )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
            joint_training (bool): Whether to perform joint training with vocoder.

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys = feats
        olens = feats_lengths

        # make labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate tacotron2 outputs
        after_outs, before_outs, logits, att_ws = self._forward(
            xs=xs, ilens=ilens, ys=ys, olens=olens, spembs=spembs, sids=sids, lids=lids,
        )

        # modify mod part of groundtruth
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

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(
            after_outs, before_outs, logits, ys, labels, olens
        )
        if self.loss_type == "L1+L2":
            loss = l1_loss + mse_loss + bce_loss
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        stats = dict(
            l1_loss=l1_loss.item(), mse_loss=mse_loss.item(), bce_loss=bce_loss.item(),
        )

        # calculate attention loss
        if self.use_guided_attn_loss:
            # NOTE(kan-bayashi): length of output for auto-regressive
            # input will be changed when r > 1
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            attn_loss = self.attn_loss(att_ws, ilens, olens_in)
            loss = loss + attn_loss
            stats.update(attn_loss=attn_loss.item())

        if not joint_training:
            stats.update(loss=loss.item())
            loss, stats, weight = force_gatherable(
                (loss, stats, batch_size), loss.device
            )
            return loss, stats, weight
        else:
            return loss, stats, after_outs

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: torch.Tensor,
        olens: torch.Tensor,
        spembs: torch.Tensor,
        sids: torch.Tensor,
        lids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hs, hlens = self.enc(xs, ilens)
        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)
        return self.dec(hs, hlens, ys)

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor]): Feature sequence to extract style (N, idim).
            spembs (Optional[Tensor]): Speaker embedding (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lids (Optional[Tensor]): Language ID (1,).
            threshold (float): Threshold in inference.
            minlenratio (float): Minimum length ratio in inference.
            maxlenratio (float): Maximum length ratio in inference.
            use_att_constraint (bool): Whether to apply attention constraint.
            backward_window (int): Backward window in attention constraint.
            forward_window (int): Forward window in attention constraint.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * prob (Tensor): Output sequence of stop probabilities (T_feats,).
                * att_w (Tensor): Attention weights (T_feats, T).

        """
        x = text
        y = feats
        spemb = spembs

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference with teacher forcing
        if use_teacher_forcing:
            assert feats is not None, "feats must be provided with teacher forcing."

            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            spembs = None if spemb is None else spemb.unsqueeze(0)
            ilens = x.new_tensor([xs.size(1)]).long()
            olens = y.new_tensor([ys.size(1)]).long()
            outs, _, _, att_ws = self._forward(
                xs=xs,
                ilens=ilens,
                ys=ys,
                olens=olens,
                spembs=spembs,
                sids=sids,
                lids=lids,
            )

            return dict(feat_gen=outs[0], att_w=att_ws[0])

        # inference
        h = self.enc.inference(x)
        if self.use_gst:
            style_emb = self.gst(y.unsqueeze(0))
            h = h + style_emb
        if self.spks is not None:
            sid_emb = self.sid_emb(sids.view(-1))
            h = h + sid_emb
        if self.langs is not None:
            lid_emb = self.lid_emb(lids.view(-1))
            h = h + lid_emb
        if self.spk_embed_dim is not None:
            hs, spembs = h.unsqueeze(0), spemb.unsqueeze(0)
            h = self._integrate_with_spk_embed(hs, spembs)[0]
        out, prob, att_w = self.dec.inference(
            h,
            threshold=threshold,
            minlenratio=minlenratio,
            maxlenratio=maxlenratio,
            use_att_constraint=use_att_constraint,
            backward_window=backward_window,
            forward_window=forward_window,
        )

        return dict(feat_gen=out, prob=prob, att_w=att_w)

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, eunits).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, eunits) if
                integration_type is "add" else (B, Tmax, eunits + spk_embed_dim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        else:
            raise NotImplementedError("support only add or concat.")

        return hs
