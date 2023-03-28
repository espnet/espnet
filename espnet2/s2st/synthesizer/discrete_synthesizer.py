# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Translatotron Synthesizer related modules for ESPnet2."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.s2st.synthesizer.abs_synthesizer import AbsSynthesizer
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.scorer_interface import BatchScorerInterface


class TransformerDiscreteSynthesizer(AbsSynthesizer, BatchScorerInterface):
    """Discrete unit Synthesizer related modules for speech-to-speech translation.

    This is a module of discrete unit prediction network in discrete-unit described
    in `Direct speech-to-speech translation with discrete units`_,
    which converts the sequence of hidden states into the sequence of discrete unit (from SSLs).

    .. _`Direct speech-to-speech translation with discrete units`:
       https://arxiv.org/abs/2107.05604

    """

    def __init__(
        self,
        # decoder related
        odim: int,
        idim: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        layer_drop_rate: float = 0.0,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "concat",
    ):
        """Transfomer decoder for discrete unit module.

        Args:
            vocab_size: output dim
            encoder_output_size: dimension of attention
            attention_heads: the number of heads of multi head attention
            linear_units: the number of units of position-wise feed forward
            num_blocks: the number of decoder blocks
            dropout_rate: dropout rate
            self_attention_dropout_rate: dropout rate for attention
            input_layer: input layer type
            use_output_layer: whether to use output layer
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            normalize_before: whether to use layer_norm before the first block
            concat_after: whether to concat attention layer's input and output
                if True, additional linear will be applied.
                i.e. x -> x + linear(concat(x, att(x)))
                if False, no additional linear will be applied.
                i.e. x -> x + att(x)
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type (str): How to integrate speaker embedding.
        """
        assert check_argument_types()
        super().__init__()

        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, encoder_output_size)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, encoder_output_size)

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
            self.projection = torch.nn.Linear(self.spk_embed_dim, encoder_output_size)
        else:
            raise ValueError(f"{spk_embed_integration_type} is not supported.")

        self.decoder = TransformerDecoder(
            vocab_size=odim,
            encoder_output_size=dec_idim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            self_attention_dropout_rate=self_attention_dropout_rate,
            src_attention_dropout_rate=src_attention_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            concat_after=concat_after,
            layer_drop_rate=layer_drop_rate,
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
        return_hs: bool = False,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            enc_outputs (LongTensor): Batch of padded character ids (B, T, idim).
            enc_outputs_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).

        Returns:
            hs
            hlens
        """

        enc_outputs = enc_outputs[:, : enc_outputs_lengths.max()]
        feats = feats[:, : feats_lengths.max()]  # for data-parallel

        ys = feats
        olens = feats_lengths

        # calculate hidden spaces for discrete unit outputs
        hs, hlens = self._forward(
            hs=enc_outputs,
            hlens=enc_outputs_lengths,
            ys=ys,
            olens=olens,
            spembs=spembs,
            sids=sids,
            lids=lids,
            return_hs=return_hs,
            return_all_hs=return_all_hs,
        )

        return hs, hlens

    def _forward(
        self,
        hs: torch.Tensor,
        hlens: torch.Tensor,
        ys: torch.Tensor,
        olens: torch.Tensor,
        spembs: torch.Tensor,
        sids: torch.Tensor,
        lids: torch.Tensor,
        return_hs: bool = False,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)
        return self.decoder(
            hs,
            hlens,
            ys,
            olens,
            return_hs=return_hs,
            return_all_hs=return_all_hs,
        )

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

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        # FIXME(jiatong): the spk/lang embedding may be execute too many times
        # consider add before the search
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            memory = memory + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            memory = memory + lid_embs.unsqueeze(1)
        if self.spk_embed_dim is not None:
            memory = self._integrate_with_spk_embed(memory, spembs)

        return self.decoder.forward_one_step(tgt, tgt_mask, memory, cache)

    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoder.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

    def inference(self):
        pass
