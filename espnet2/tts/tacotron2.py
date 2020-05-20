# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules."""

import logging
from typing import Dict
from typing import Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import GuidedAttentionLoss
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2Loss
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.attentions import AttForward
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.rnn.attentions import AttLoc
from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHG
from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHGLoss
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.abs_tts import AbsTTS


class Tacotron2(AbsTTS):
    """Tacotron2 module for end-to-end text-to-speech.

    This is a module of Spectrogram prediction network in Tacotron2 described
    in `Natural TTS Synthesis
    by Conditioning WaveNet on Mel Spectrogram Predictions`_, which converts
    the sequence of characters into the sequence of Mel-filterbanks.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    Args:
        idim: Dimension of the inputs.
        odim: Dimension of the outputs.
        spk_embed_dim: Dimension of the speaker embedding.
        embed_dim: Dimension of character embedding.
        elayers: The number of encoder blstm layers.
        eunits: The number of encoder blstm units.
        econv_layers: The number of encoder conv layers.
        econv_filts: The number of encoder conv filter size.
        econv_chans: The number of encoder conv filter channels.
        dlayers: The number of decoder lstm layers.
        dunits: The number of decoder lstm units.
        prenet_layers: The number of prenet layers.
        prenet_units: The number of prenet units.
        postnet_layers: The number of postnet layers.
        postnet_filts: The number of postnet filter size.
        postnet_chans: The number of postnet filter channels.
        output_activation: The name of activation function for outputs.
        adim: The number of dimension of mlp in attention.
        aconv_chans: The number of attention conv filter channels.
        aconv_filts: The number of attention conv filter size.
        cumulate_att_w: Whether to cumulate previous attention weight.
        use_batch_norm: Whether to use batch normalization.
        use_concate: Whether to concatenate encoder embedding with decoder
            lstm outputs.
        dropout_rate: Dropout rate.
        zoneout_rate: Zoneout rate.
        reduction_factor: Reduction factor.
        spk_embed_dim: Number of speaker embedding dimenstions.
        spc_dim: Number of spectrogram embedding dimenstions
            (only for use_cbhg=True).
        use_cbhg: Whether to use CBHG module.
        cbhg_conv_bank_layers: The number of convoluional banks in CBHG.
        cbhg_conv_bank_chans: The number of channels of convolutional bank in
            CBHG.
        cbhg_proj_filts: The number of filter size of projection layeri in
            CBHG.
        cbhg_proj_chans: The number of channels of projection layer in CBHG.
        cbhg_highway_layers: The number of layers of highway network in CBHG.
        cbhg_highway_units: The number of units of highway network in CBHG.
        cbhg_gru_units: The number of units of GRU in CBHG.
        use_masking: Whether to mask padded part in loss calculation.
        use_weighted_masking: Whether to apply weighted masking in
            loss calculation.
        bce_pos_weight: Weight of positive sample of stop token
            (only for use_masking=True).
        use_guided_attn_loss: Whether to use guided attention loss.
        guided_attn_loss_sigma: Sigma in guided attention loss.
        guided_attn_loss_lamdba: Lambda in guided attention loss.
    """

    def __init__(
        self,
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
        use_cbhg: bool = False,
        cbhg_conv_bank_layers: int = 8,
        cbhg_conv_bank_chans: int = 128,
        cbhg_conv_proj_filts: int = 3,
        cbhg_conv_proj_chans: int = 256,
        cbhg_highway_layers: int = 4,
        cbhg_highway_units: int = 128,
        cbhg_gru_units: int = 256,
        use_batch_norm: bool = True,
        use_concate: bool = True,
        use_residual: bool = False,
        dropout_rate: float = 0.5,
        zoneout_rate: float = 0.1,
        reduction_factor: int = 1,
        spk_embed_dim: int = None,
        spc_dim: int = None,
        use_masking: bool = True,
        use_weighted_masking: bool = False,
        bce_pos_weight: float = 5.0,
        use_guided_attn_loss: bool = True,
        guided_attn_loss_sigma: float = 0.4,
        guided_attn_loss_lambda: float = 1.0,
    ):
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
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

        dec_idim = eunits if spk_embed_dim is None else eunits + spk_embed_dim
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
        if self.use_cbhg:
            self.cbhg = CBHG(
                idim=odim,
                odim=spc_dim,
                conv_bank_layers=cbhg_conv_bank_layers,
                conv_bank_chans=cbhg_conv_bank_chans,
                conv_proj_filts=cbhg_conv_proj_filts,
                conv_proj_chans=cbhg_conv_proj_chans,
                highway_layers=cbhg_highway_layers,
                highway_units=cbhg_highway_units,
                gru_units=cbhg_gru_units,
            )
            self.cbhg_loss = CBHGLoss(use_masking=use_masking)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor = None,
        spcs: torch.Tensor = None,
        spcs_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text: Batch of padded character ids (B, Tmax).
            text_lengths: Batch of lengths of each input batch (B,).
            speech: Batch of padded target features (B, Lmax, odim).
            speech_lengths: Batch of the lengths of each target (B,).
            spembs: Batch of speaker embedding vectors (B, spk_embed_dim).
            spcs: Batch of ground-truth spectrogram (B, Lmax, spc_dim).
            spcs_lengths:
        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        speech = speech[:, : speech_lengths.max()]  # for data-parallel

        batch_size = text.size(0)
        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", 0.0)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys = speech
        olens = speech_lengths

        # make labels for stop prediction
        labels = make_pad_mask(olens).to(ys.device, ys.dtype)

        # calculate tacotron2 outputs
        hs, hlens = self.enc(xs, ilens)
        if self.spk_embed_dim is not None:
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        after_outs, before_outs, logits, att_ws = self.dec(hs, hlens, ys)

        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_out = max(olens)
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]
            labels[:, -1] = 1.0  # make sure at least one frame has 1

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(
            after_outs, before_outs, logits, ys, labels, olens
        )
        loss = l1_loss + mse_loss + bce_loss

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

        # caluculate cbhg loss
        if self.use_cbhg:
            # remove unnecessary padded part (for multi-gpus)
            if max_out != spcs.shape[1]:
                spcs = spcs[:, :max_out]

            # caluculate cbhg outputs & loss and report them
            cbhg_outs, _ = self.cbhg(after_outs, olens)
            cbhg_l1_loss, cbhg_mse_loss = self.cbhg_loss(cbhg_outs, spcs, olens)
            loss = loss + cbhg_l1_loss + cbhg_mse_loss
            stats.update(
                cbhg_l1_loss=cbhg_l1_loss.item(), cbhg_mse_loss=cbhg_mse_loss.item(),
            )

        stats.update(loss=loss.item())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference(
        self,
        text: torch.Tensor,
        spembs: torch.Tensor = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text: Input sequence of characters (T,).
            spembs: Speaker embedding vector (spk_embed_dim,).
            threshold: Threshold in inference.
            minlenratio: Minimum length ratio in inference.
            maxlenratio: Maximum length ratio in inference.
            use_att_constraint: Whether to apply attention constraint.
            backward_window: Backward window in attention constraint.
            forward_window: Forward window in attention constraint.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).

        """
        x = text
        spemb = spembs

        # inference
        h = self.enc.inference(x)
        if self.spk_embed_dim is not None:
            spemb = F.normalize(spemb, dim=0).unsqueeze(0).expand(h.size(0), -1)
            h = torch.cat([h, spemb], dim=-1)
        outs, probs, att_ws = self.dec.inference(
            h,
            threshold=threshold,
            minlenratio=minlenratio,
            maxlenratio=maxlenratio,
            use_att_constraint=use_att_constraint,
            backward_window=backward_window,
            forward_window=forward_window,
        )

        if self.use_cbhg:
            cbhg_outs = self.cbhg.inference(outs)
            return cbhg_outs, probs, att_ws
        else:
            return outs, probs, att_ws
