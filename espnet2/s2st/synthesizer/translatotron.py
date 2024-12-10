# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Translatotron Synthesizer related modules for ESPnet2."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.s2st.synthesizer.abs_synthesizer import AbsSynthesizer
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.attentions import (
    AttForward,
    AttForwardTA,
    AttLoc,
    AttMultiHeadAdd,
)
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder


class Translatotron(AbsSynthesizer):
    """
    Translatotron Synthesizer related modules for speech-to-speech translation.

    This module is part of the Spectrogram prediction network in Translatotron
    described in `Direct speech-to-speech translation with a sequence-to-sequence
    model`_, which converts the sequence of hidden states into the sequence of
    Mel-filterbanks.

    .. _`Direct speech-to-speech translation with a sequence-to-sequence model`:
       https://arxiv.org/pdf/1904.06037.pdf

    Attributes:
        idim (int): Dimension of the inputs.
        odim (int): Dimension of the outputs.
        atype (str): Type of attention mechanism.
        cumulate_att_w (bool): Whether to cumulate previous attention weight.
        reduction_factor (int): Reduction factor for outputs.
        output_activation_fn (callable, optional): Activation function for the output.
        padding_idx (int): Index used for padding in input sequences.
        spks (Optional[int]): Number of speakers.
        langs (Optional[int]): Number of languages.
        spk_embed_dim (Optional[int]): Dimension of speaker embeddings.
        sid_emb (torch.nn.Embedding, optional): Embedding layer for speaker IDs.
        lid_emb (torch.nn.Embedding, optional): Embedding layer for language IDs.
        projection (torch.nn.Linear, optional): Linear projection for speaker embeddings.

    Args:
        idim (int): Dimension of the inputs.
        odim (int): Dimension of the outputs.
        embed_dim (int): Dimension of the token embedding (default=512).
        atype (str): Type of attention (default="multihead").
        adim (int): Number of dimensions of MLP in attention (default=512).
        aheads (int): Number of attention heads (default=4).
        aconv_chans (int): Number of attention convolution filter channels (default=32).
        aconv_filts (int): Size of attention convolution filter (default=15).
        cumulate_att_w (bool): Whether to cumulate previous attention weight (default=True).
        dlayers (int): Number of decoder LSTM layers (default=4).
        dunits (int): Number of decoder LSTM units (default=1024).
        prenet_layers (int): Number of prenet layers (default=2).
        prenet_units (int): Number of prenet units (default=32).
        postnet_layers (int): Number of postnet layers (default=5).
        postnet_chans (int): Number of postnet filter channels (default=512).
        postnet_filts (int): Size of postnet filter (default=5).
        output_activation (Optional[str]): Name of activation function for outputs.
        use_batch_norm (bool): Whether to use batch normalization (default=True).
        use_concate (bool): Whether to concatenate encoder outputs with decoder LSTM
            outputs (default=True).
        use_residual (bool): Whether to use residual connections (default=False).
        reduction_factor (int): Reduction factor (default=2).
        spks (Optional[int]): Number of speakers (default=None).
        langs (Optional[int]): Number of languages (default=None).
        spk_embed_dim (Optional[int]): Speaker embedding dimension (default=None).
        spk_embed_integration_type (str): How to integrate speaker embedding
            (default="concat").
        dropout_rate (float): Dropout rate (default=0.5).
        zoneout_rate (float): Zoneout rate (default=0.1).

    Examples:
        >>> model = Translatotron(idim=80, odim=80)
        >>> enc_outputs = torch.randn(10, 100, 80)  # Example encoder outputs
        >>> enc_outputs_lengths = torch.randint(1, 100, (10,))
        >>> feats = torch.randn(10, 50, 80)  # Example target features
        >>> feats_lengths = torch.randint(1, 50, (10,))
        >>> outputs = model(enc_outputs, enc_outputs_lengths, feats, feats_lengths)

    Raises:
        ValueError: If an unsupported activation function is provided.
        NotImplementedError: If an unsupported attention type is provided.
    """

    @typechecked
    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        embed_dim: int = 512,
        atype: str = "multihead",
        adim: int = 512,
        aheads: int = 4,
        aconv_chans: int = 32,
        aconv_filts: int = 15,
        cumulate_att_w: bool = True,
        dlayers: int = 4,
        dunits: int = 1024,
        prenet_layers: int = 2,
        prenet_units: int = 32,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        output_activation: Optional[str] = None,
        use_batch_norm: bool = True,
        use_concate: bool = True,
        use_residual: bool = False,
        reduction_factor: int = 2,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "concat",
        # training related
        dropout_rate: float = 0.5,
        zoneout_rate: float = 0.1,
    ):
        """Initialize Tacotron2 module.

        Args:
            idim (int): Dimension of the inputs.
            odim: (int) Dimension of the outputs.
            adim (int): Number of dimension of mlp in attention.
            atype (str): type of attention
            aconv_chans (int): Number of attention conv filter channels.
            aconv_filts (int): Number of attention conv filter size.
            embed_dim (int): Dimension of the token embedding.
            dlayers (int): Number of decoder lstm layers.
            dunits (int): Number of decoder lstm units.
            prenet_layers (int): Number of prenet layers.
            prenet_units (int): Number of prenet units.
            postnet_layers (int): Number of postnet layers.
            postnet_filts (int): Number of postnet filter size.
            postnet_chans (int): Number of postnet filter channels.
            output_activation (str): Name of activation function for outputs.
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
            dropout_rate (float): Dropout rate.
            zoneout_rate (float): Zoneout rate.
        """
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.atype = atype
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor

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

        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, idim)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, idim)

        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is None:
            dec_idim = idim
        elif self.spk_embed_integration_type == "concat":
            dec_idim = idim + spk_embed_dim
        elif self.spk_embed_integration_type == "add":
            dec_idim = idim
            self.projection = torch.nn.Linear(self.spk_embed_dim, idim)
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
        elif atype == "multihead":
            att = AttMultiHeadAdd(dec_idim, dunits, aheads, adim, adim)
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

    def forward(
        self,
        enc_outputs: torch.Tensor,
        enc_outputs_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate forward propagation.

        This method performs the forward pass through the Translatotron model,
        processing the encoded outputs and generating the corresponding target
        features. It also computes the attention weights and stop labels.

        Args:
            enc_outputs (LongTensor): Batch of padded character ids (B, T, idim).
            enc_outputs_lengths (LongTensor): Batch of lengths of each input
                batch (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            spembs (Optional[Tensor]): Batch of speaker embeddings
                (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - after_outs: Output features after the forward pass.
                - before_outs: Output features before the forward pass.
                - logits: Logits for stop prediction.
                - att_ws: Attention weights.
                - ys: Ground truth features.
                - labels: Labels for stop prediction.
                - olens: Lengths of output sequences.

        Note:
            The method assumes that input tensors are properly padded and that
            their lengths are provided. The maximum lengths of the inputs are
            used to slice the tensors for processing.

        Examples:
            >>> enc_outputs = torch.randn(2, 10, 80)  # Batch of 2
            >>> enc_outputs_lengths = torch.tensor([10, 8])
            >>> feats = torch.randn(2, 20, 80)  # Batch of 2
            >>> feats_lengths = torch.tensor([20, 18])
            >>> after_outs, before_outs, logits, att_ws, ys, labels, olens =
            ...     model.forward(enc_outputs, enc_outputs_lengths, feats,
            ...     feats_lengths)

        Raises:
            ValueError: If the provided activation function name is invalid.
        """

        enc_outputs = enc_outputs[:, : enc_outputs_lengths.max()]
        feats = feats[:, : feats_lengths.max()]  # for data-parallel

        ys = feats
        olens = feats_lengths

        # make labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate tacotron2 outputs
        after_outs, before_outs, logits, att_ws = self._forward(
            hs=enc_outputs,
            hlens=enc_outputs_lengths,
            ys=ys,
            olens=olens,
            spembs=spembs,
            sids=sids,
            lids=lids,
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

        return after_outs, before_outs, logits, att_ws, ys, labels, olens

    def _forward(
        self,
        hs: torch.Tensor,
        hlens: torch.Tensor,
        ys: torch.Tensor,
        olens: torch.Tensor,
        spembs: torch.Tensor,
        sids: torch.Tensor,
        lids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        enc_outputs: torch.Tensor,
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
            enc_outputs (LongTensor): Input sequence of characters (N, idim).
            feats (Optional[Tensor]): Feature sequence to extract style (N, odim).
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
        h = enc_outputs
        y = feats
        spemb = spembs

        # inference with teacher forcing
        if use_teacher_forcing:
            assert feats is not None, "feats must be provided with teacher forcing."

            hs, ys = h.unsqueeze(0), y.unsqueeze(0)
            spembs = None if spemb is None else spemb.unsqueeze(0)
            hlens = h.new_tensor([hs.size(1)]).long()
            olens = y.new_tensor([ys.size(1)]).long()
            outs, _, _, att_ws = self._forward(
                hs=hs,
                hlens=hlens,
                ys=ys,
                olens=olens,
                spembs=spembs,
                sids=sids,
                lids=lids,
            )

            return dict(feat_gen=outs[0], att_w=att_ws[0])

        # inference
        if self.spks is not None:
            sid_emb = self.sid_emb(sids.view(-1))
            enc_outputs = enc_outputs + sid_emb
        if self.langs is not None:
            lid_emb = self.lid_emb(lids.view(-1))
            enc_outputs = enc_outputs + lid_emb
        if self.spk_embed_dim is not None:
            enc_outputs, spembs = enc_outputs.unsqueeze(0), spemb.unsqueeze(0)
            enc_outputs = self._integrate_with_spk_embed(enc_outputs, spembs)[0]
        out, prob, att_w = self.dec.inference(
            enc_outputs,
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
