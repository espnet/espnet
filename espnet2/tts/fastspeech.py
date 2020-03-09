#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FastSpeech related modules."""

import logging
from typing import Dict
from typing import Tuple
from typing import Sequence

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.nets.pytorch_backend.e2e_tts_transformer import TTSPlot
from espnet.nets.pytorch_backend.fastspeech.duration_calculator import DurationCalculator
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictorLoss
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.e2e_tts_fastspeech import FeedForwardTransformerLoss
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.abs_model import AbsTTS


class FeedForwardTransformer(AbsTTS):
    """Feed Forward Transformer for TTS a.k.a. FastSpeech.

    This is a module of FastSpeech, feed-forward Transformer with duration predictor described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_, which does not require any auto-regressive
    processing during inference, resulting in fast decoding compared with auto-regressive Transformer.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    
    Args:
        TODO
    """
    def __init__(
        self,
        idim: int,
        odim: int,
        adim: int = 384,
        aheads: int = 2,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 256,
        duration_predictor_kernel_size: int = 3,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 3,
        postnet_layers: int = 5,
        postnet_filts: int = 5,
        postnet_chans: int = 256,
        use_batch_norm: bool = True,
        use_scaled_pos_enc: bool = True,
        encoder_normalize_before: bool = False,
        decoder_normalize_before: bool = False,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        reduction_factor: int = 1,
        spk_embed_dim: int = None,        # speaker embedding dimension
        spk_embed_integration_type: str = "add",
        initial_encoder_alpha: float = 1.0,
        initial_decoder_alpha: float = 1.0,
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        transformer_enc_dec_attn_dropout_rate: float = 0.1,
        postnet_dropout_rate: float = 0.5,
        duration_predictor_dropout_rate: float = 0.1,
        transfer_encoder_from_teacher: bool = False,
        transferred_encoder_module: str = "embed",
        use_masking: bool = True,
        use_weighted_masking: bool = False,
        transformer_init: str = "pytorch",
        teacher_model: str = None,
    ):
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.spk_embed_dim = spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.reduction_factor = reduction_factor

        # set padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim,
            embedding_dim=adim,
            padding_idx=padding_idx
        )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=encoder_input_layer,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size
        )

        # define additional projection for speaker embedding
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder because fastspeech's decoder is the same as encoder
        self.decoder = Encoder(
            idim=0,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            input_layer=None,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            attention_dropout_rate=transformer_dec_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size
        )

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        # define postnet
        self.postnet = None if postnet_layers == 0 else Postnet(
            idim=idim,
            odim=odim,
            n_layers=postnet_layers,
            n_chans=postnet_chans,
            n_filts=postnet_filts,
            use_batch_norm=use_batch_norm,
            dropout_rate=postnet_dropout_rate
        )

        # initialize parameters
        self._reset_parameters(init_type=transformer_init,
                               init_enc_alpha=initial_encoder_alpha,
                               init_dec_alpha=initial_decoder_alpha)

         # define teacher model
        if teacher_model is not None:
            self.teacher = self._load_teacher_model(teacher_model)
        else:
            self.teacher = None

        # define duration calculator
        if self.teacher is not None:
            self.duration_calculator = DurationCalculator(self.teacher)
        else:
            self.duration_calculator = None

        # transfer teacher parameters
        if self.teacher is not None and transfer_encoder_from_teacher:
            self._transfer_from_teacher(transferred_encoder_module)

        # define criterions
        self.criterion = FeedForwardTransformerLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking
        )

    def _forward(self, xs, ilens, ys=None, olens=None, spembs=None, ds=None, is_inference=False):
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(ilens).to(xs.device)
        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, d_outs, ilens)  # (B, Lmax, adim)
        else:
            if ds is None:
                assert self.duration_calculator is not None, \
                    "Need an auto-regressive teacher model to calculate durations."
                with torch.no_grad():
                    ds = self.duration_calculator(xs, ilens, ys, olens, spembs)  # (B, Tmax)
            d_outs = self.duration_predictor(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, ds, ilens)  # (B, Lmax, adim)

        # forward decoder
        if olens is not None:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        if is_inference:
            return before_outs, after_outs, d_outs
        else:
            return before_outs, after_outs, ds, d_outs

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        ds: torch.Tensor,
        spembs: torch.Tensor = None,
        spcs: torch.Tensor = None,
        spcs_lengths: torch.Tensor = None,
        is_inference: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text: Batch of padded character ids (B, Tmax).
            text_lengths: Batch of lengths of each input batch (B,).
            speech: Batch of padded target features (B, Lmax, odim).
            speech_lengths: Batch of the lengths of each target (B,).
            ds: Batch of precalculated durations (B, Tmax, 1).
            spembs: Batch of speaker embedding vectors (B, spk_embed_dim).
            spcs: Batch of ground-truth spectrogram (B, Lmax, spc_dim).
            spcs_lengths: Batch of the lengths of each spc (B,).
        """

        text = text[:, : text_lengths.max()]  # for data-parallel
        speech = speech[:, : speech_lengths.max()]  # for data-parallel

        batch_size = text.size(0)
        # TODO: need to append <eos> to text?
        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", 0.0)
        for i, l in enumerate(text_lengths):
           xs[i, l] = self.eos
        ilens = text_lengths + 1
        # xs = text
        # ilens = text_lengths

        ys = speech
        olens = speech_lengths

        if ds is not None:
            ds = ds[:, :max(ilens)].squeeze(-1)

        # forward propagation
        before_outs, after_outs, ds, d_outs = self._forward(
            xs, ilens, ys, olens, spembs=spembs, ds=ds, is_inference=False)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        # calculate loss
        if self.postnet is None:
            l1_loss, duration_loss = self.criterion(None, before_outs, d_outs, ys, ds, ilens, olens)
        else:
            l1_loss, duration_loss = self.criterion(after_outs, before_outs, d_outs, ys, ds, ilens, olens)
        loss = l1_loss + duration_loss

        stats = dict(
            loss=loss.item(), l1_loss=l1_loss.item(), duration_loss=duration_loss.item()
        )

        # report extra information
        if self.use_scaled_pos_enc:
            stats.update(encoder_alpha=self.encoder.embed[-1].alpha.data.item())
            stats.update(decoder_alpha=self.decoder.embed[-1].alpha.data.item())

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
            None: Dummy for compatibility.
            None: Dummy for compatibility.

        """
        x = text
        spemb = spembs

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)
        if spemb is not None:
            spembs = spemb.unsqueeze(0)
        else:
            spembs = None

        # inference
        _, outs, _ = self._forward(xs, ilens, spembs=spembs, is_inference=True)  # (1, L, odim)

        return outs[0], None, None

    def _integrate_with_spk_embed(self, hs, spembs):
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim)

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs
    
    def _source_mask(self, ilens):
        """Make masks for self-attention.

        Args:
            ilens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _load_teacher_model(self, model_path):
        # NOTE: This is not tested yet.
        # get teacher model config
        idim, odim, args = get_model_conf(model_path)  #TODO

        # assert dimension is the same between teacher and studnet
        assert idim == self.idim
        assert odim == self.odim
        assert args.reduction_factor == self.reduction_factor

        # load teacher model
        from espnet.utils.dynamic_import import dynamic_import
        model_class = dynamic_import(args.model_module)  #TODO
        model = model_class(idim, odim, args)
        torch_load(model_path, model)

        # freeze teacher model parameters
        for p in model.parameters():
            p.requires_grad = False

        return model    

    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

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



















