"""ESPnet wrapper for the NeMo-free Sortformer speaker-diarization model.

NeMo-free reimplementation of NVIDIA's ``SortformerEncLabelModel`` as an
:class:`~espnet2.train.abs_espnet_model.AbsESPnetModel`. The pipeline is::

    audio -> peak-normalize -> log-mel (MelSpectrogramPreprocessor)
          -> FastConformerEncoder (8x subsampling)
          -> encoder_proj (512 -> 192)
          -> TransformerEncoder (post-LN)
          -> sigmoid speaker head -> per-frame speaker probabilities (B, T, 4)

Both offline (full-context) and streaming long-form inference are supported. The
streaming path carries the :class:`SortformerModules` speaker cache across chunks
so output channel ``k`` keeps the same speaker over the whole session; with the
default chunk size (~15 s) a 90 s recording is processed as ~6 chunks.

Training uses the hybrid Arrival-Time-Sort + Permutation-Invariant BCE loss
(:class:`~sortformer.sort_loss.SortformerHybridLoss`); with
``train_streaming=True`` the predictions come from the streaming cache loop
(BPTT through the cache).

The component sub-modules mirror the released ``nvidia/diar_sortformer_4spk-v1``
checkpoint so the weights can be converted with a near-identity key remap.
"""

import math
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch

from espnet2.train.abs_espnet_model import AbsESPnetModel

from .fastconformer_encoder import FastConformerEncoder
from .preprocessor import MelSpectrogramPreprocessor
from .sort_loss import SortformerHybridLoss
from .sortformer_modules import SortformerModules
from .transformer_encoder import TransformerEncoder


@contextmanager
def _null_context():
    yield


class ESPnetSortformerModel(AbsESPnetModel):
    """NeMo-free Sortformer speaker-diarization model (4 speakers by default).

    Wraps the full Sortformer stack as an
    :class:`~espnet2.train.abs_espnet_model.AbsESPnetModel`. The forward path is
    ``preprocessor -> FastConformer encoder -> encoder_proj (fc_d_model ->
    tf_d_model) -> Transformer encoder -> sigmoid speaker head``, producing
    per-frame speaker probabilities of shape ``(B, T, num_spk)``.

    Two inference modes are provided:

    * Offline (full context): :meth:`encode` / :meth:`diarize` run the whole
      utterance through the encoder at once.
    * Streaming (long-form): :meth:`forward_streaming` /
      :meth:`diarize_streaming` process the audio chunk by chunk while carrying
      the :class:`~sortformer.sortformer_modules.SortformerModules` speaker
      cache across chunks, keeping output channel ``k`` bound to the same
      speaker over the whole session. With the default streaming
      hyper-parameters (chunk ~15 s) a 90 s session is processed as ~6 chunks.

    Training (:meth:`forward`) uses the hybrid Arrival-Time-Sort +
    Permutation-Invariant BCE loss
    (:class:`~sortformer.sort_loss.SortformerHybridLoss`). When
    ``train_streaming=True`` the training predictions are produced by the
    streaming cache loop (BPTT through the cache) instead of the offline
    full-context :meth:`encode`, matching the original Sortformer training
    regime.

    Args:
        preprocessor: Waveform -> log-mel feature extractor.
        encoder: FastConformer encoder (``subsampling_factor`` of 8 by default).
        sortformer_modules: Holds ``encoder_proj``, the sigmoid speaker head and
            the streaming speaker-cache machinery.
        transformer_encoder: Post-LN Transformer applied on top of the
            projected encoder states.
        num_spk: Number of output speaker channels.
        ats_weight: Weight of the Arrival-Time-Sort term in the hybrid loss.
        pil_weight: Weight of the Permutation-Invariant term in the hybrid loss.
        eps: Stabilizer for the peak normalization of the waveform.
        train_streaming: If True, :meth:`forward` runs the streaming cache loop
            to produce training predictions instead of the offline encoder.

    Example:
        >>> model = build_sortformer_model(num_spk=4)
        >>> preds, lengths = model.diarize(speech, speech_lengths)
        >>> preds.shape  # (B, T, num_spk) sigmoid speaker activities
    """

    def __init__(
        self,
        preprocessor: MelSpectrogramPreprocessor,
        encoder: FastConformerEncoder,
        sortformer_modules: SortformerModules,
        transformer_encoder: TransformerEncoder,
        num_spk: int = 4,
        ats_weight: float = 0.5,
        pil_weight: float = 0.5,
        eps: float = 1e-3,
        train_streaming: bool = False,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.sortformer_modules = sortformer_modules
        self.transformer_encoder = transformer_encoder
        self.num_spk = num_spk
        self.eps = eps
        # When True, training predictions are produced by the streaming speaker
        # cache (forward_streaming, cache in the loop) -- as in original
        # Sortformer training -- instead of the offline full-context encode().
        self.train_streaming = train_streaming
        self.loss = SortformerHybridLoss(
            num_spks=num_spk, ats_weight=ats_weight, pil_weight=pil_weight
        )

    # ------------------------------------------------------------------ #
    # core forward
    # ------------------------------------------------------------------ #
    def _process_signal(self, speech, speech_lengths):
        """Peak-normalize the waveform and extract log-mel features.

        Mirrors NeMo's ``process_signal``: the waveform is scaled by the global
        max over the batch tensor (with ``eps`` for stability) before feature
        extraction. Returns ``(B, n_mels, T_feat)`` features and their lengths.
        """
        speech = (1.0 / (speech.max() + self.eps)) * speech
        feats, feat_lengths = self.preprocessor(speech, speech_lengths)
        feats = feats[:, :, : feat_lengths.max()]
        return feats, feat_lengths  # (B, n_mels, T_feat)

    def frontend_encoder(
        self, x, lengths, bypass_pre_encode: bool = False, full_prefix_len: int = 0
    ):
        """Run the FastConformer encoder and project to the Transformer width.

        Args:
            x: ``(B, n_mels, T)`` log-mel features when
                ``bypass_pre_encode=False``, or ``(B, T', fc_d_model)``
                pre-encoded embeddings when ``bypass_pre_encode=True`` (the
                streaming path, where the speaker cache and FIFO are already in
                pre-encode space).
            lengths: Valid length per item, in the units of ``x``.
            bypass_pre_encode: Skip the convolutional sub-sampling front end and
                treat ``x`` as already pre-encoded.
            full_prefix_len: Streaming only. Number of leading speaker-cache +
                FIFO frames that must always be attendable under the encoder's
                local attention.

        Returns:
            Tuple of projected encoder states ``(B, T, tf_d_model)`` and their
            lengths.
        """
        if not bypass_pre_encode:
            x = x.transpose(1, 2)  # (B, T_feat, n_mels)
        emb, emb_lengths, _ = self.encoder(
            x,
            lengths,
            bypass_pre_encode=bypass_pre_encode,
            full_prefix_len=full_prefix_len,
        )
        emb = self.sortformer_modules.encoder_proj(emb)
        return emb, emb_lengths

    def forward_infer(self, emb, emb_lengths, n_global: int = 0):
        """Projected encoder states -> masked speaker probabilities.

        Applies the Transformer encoder and the sigmoid speaker head, then zeros
        out predictions beyond each item's valid length.

        Args:
            emb: Projected encoder states ``(B, T, tf_d_model)``.
            emb_lengths: Valid length per item.
            n_global: Streaming only. Number of leading global (cache + FIFO)
                frames forwarded to the Transformer.

        Returns:
            Speaker probabilities ``(B, T, num_spk)`` with padded frames zeroed.
        """
        trans = self.transformer_encoder(emb, emb_lengths, n_global=n_global)
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans)
        max_len = preds.size(1)
        valid = torch.arange(max_len, device=preds.device).expand(
            preds.size(0), max_len
        ) < emb_lengths.unsqueeze(1)
        return preds * valid.unsqueeze(-1)

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Offline full-context encode of an utterance.

        Runs ``preprocessor -> frontend_encoder -> forward_infer`` over the whole
        input with no speaker cache. Use :meth:`forward_streaming` for long-form
        audio that needs session-consistent speaker channels.

        Args:
            speech: Waveform ``(B, n_samples)``.
            speech_lengths: Valid samples per item.

        Returns:
            Tuple of speaker probabilities ``(B, T, num_spk)`` and the
            per-frame output lengths.
        """
        feats, feat_lengths = self._process_signal(speech, speech_lengths)
        emb, emb_lengths = self.frontend_encoder(
            feats, feat_lengths, bypass_pre_encode=False
        )
        preds = self.forward_infer(emb, emb_lengths)
        return preds, emb_lengths

    # ------------------------------------------------------------------ #
    # streaming (long-form) inference with the speaker cache
    # ------------------------------------------------------------------ #
    def forward_streaming(self, feats, feat_lengths):
        """Chunk-wise streaming diarization with the speaker cache.

        Iterates over fixed-size chunks (``streaming_feat_loader``) and, for each
        chunk, pre-encodes it and concatenates it after the running speaker cache
        and FIFO (``[spkcache, fifo, chunk]``). The whole concatenation is fed
        through the encoder/Transformer/head, so the cache provides global
        speaker context and output channel ``k`` keeps the same speaker across
        chunks. After each chunk the cache and FIFO are advanced by
        ``streaming_update`` and only the chunk's own predictions are appended to
        the output. With the default hyper-parameters a 90 s session is ~6
        chunks. This path is differentiable and is reused for streaming training
        (``train_streaming=True``).

        Args:
            feats: Log-mel features ``(B, n_mels, T_feat)``.
            feat_lengths: Valid feature frames per item.

        Returns:
            Concatenated per-frame speaker probabilities ``(B, T_diar,
            num_spk)`` over the whole session.
        """
        mods = self.sortformer_modules
        b = feats.shape[0]
        device = feats.device
        state = mods.init_streaming_state(batch_size=b, device=device)
        offset = torch.zeros((b,), dtype=torch.long, device=device)
        total_preds = torch.zeros((b, 0, mods.n_spk), device=device)
        sub = self.encoder.subsampling_factor
        for _, chunk_feat, flen, left, right in mods.streaming_feat_loader(
            feats, feat_lengths, offset
        ):
            chunk_pre, chunk_pre_len = self.encoder.pre_encode(chunk_feat, flen)
            concat = mods.concat_embs(
                [state.spkcache, state.fifo, chunk_pre], dim=1, device=device
            )
            prefix_len = state.spkcache.shape[1] + state.fifo.shape[1]
            concat_len = prefix_len + chunk_pre_len
            emb, emb_len = self.frontend_encoder(
                concat, concat_len, bypass_pre_encode=True, full_prefix_len=prefix_len
            )
            preds = self.forward_infer(emb, emb_len, n_global=prefix_len)
            preds = mods.apply_mask_to_preds(preds, emb_len)
            state, chunk_preds = mods.streaming_update(
                state,
                chunk_pre,
                preds,
                lc=round(left / sub),
                rc=math.ceil(right / sub),
            )
            total_preds = torch.cat([total_preds, chunk_preds], dim=1)
        return total_preds

    @torch.no_grad()
    def diarize_streaming(
        self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Long-form streaming inference entry point.

        Sets eval mode, extracts features and runs :meth:`forward_streaming`
        under ``no_grad``. Prefer this over :meth:`diarize` for sessions longer
        than a single chunk, since the speaker cache keeps the output speaker
        channels consistent across the whole recording.

        Args:
            speech: Waveform ``(B, n_samples)``.
            speech_lengths: Valid samples per item; defaults to the full length.

        Returns:
            Tuple of speaker probabilities ``(B, T, num_spk)`` and per-item
            output lengths.
        """
        if speech_lengths is None:
            speech_lengths = speech.new_full(
                (speech.size(0),), speech.size(1), dtype=torch.long
            )
        self.eval()
        feats, feat_lengths = self._process_signal(speech, speech_lengths)
        total_preds = self.forward_streaming(feats, feat_lengths)
        lengths = total_preds.new_full(
            (total_preds.size(0),), total_preds.size(1), dtype=torch.long
        )
        return total_preds, lengths

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: Optional[torch.Tensor] = None,
        spk_labels: Optional[torch.Tensor] = None,
        spk_labels_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Training step: compute the hybrid ATS + PIL loss.

        Produces per-frame speaker predictions, aligns the reference speaker
        labels to them, and evaluates the hybrid loss plus a frame-level F1.
        The prediction path depends on ``train_streaming``:

        * ``False`` (default): offline full-context :meth:`encode`.
        * ``True``: the streaming cache loop (:meth:`forward_streaming`), with
          gradients flowing through the cache.

        Args:
            speech: Waveform ``(B, n_samples)``.
            speech_lengths: Valid samples per item; defaults to the full length.
            spk_labels: Reference speaker activities ``(B, T_lab, num_spk)``.
                Required.
            spk_labels_lengths: Valid label frames per item; defaults to the
                prediction lengths.
            **kwargs: Ignored extra batch fields.

        Returns:
            Tuple of ``(loss, stats, weight)`` as expected by the ESPnet
            trainer, where ``stats`` holds ``loss``, ``ats_loss``, ``pil_loss``
            and ``f1_acc``.

        Raises:
            AssertionError: If ``spk_labels`` is None.
        """
        if speech_lengths is None:
            speech_lengths = speech.new_full(
                (speech.size(0),), speech.size(1), dtype=torch.long
            )
        assert spk_labels is not None, "spk_labels are required for training"

        if self.train_streaming:
            # Streaming training: predictions come from the speaker cache (BPTT
            # through the cache), matching original Sortformer training.
            feats, feat_lengths = self._process_signal(speech, speech_lengths)
            preds = self.forward_streaming(feats, feat_lengths)
            t_pred = preds.size(1)
            # Per-item valid diar frames from sample lengths (8x subsampling).
            preds_lengths = (
                torch.div(
                    speech_lengths.float(),
                    self.preprocessor.hop_length * self.encoder.subsampling_factor,
                    rounding_mode="floor",
                )
                .long()
                .clamp(max=t_pred)
            )
        else:
            preds, preds_lengths = self.encode(speech, speech_lengths)

        # Align targets (B, T_lab, S) to predictions (B, T_pred, S).
        targets, target_lens = self._align_targets(
            preds, preds_lengths, spk_labels, spk_labels_lengths
        )

        loss, ats_loss, pil_loss = self.loss(preds, targets, target_lens)

        with torch.no_grad():
            f1 = self._frame_f1(preds, targets, target_lens)
        stats = dict(
            loss=loss.detach(),
            ats_loss=ats_loss.detach(),
            pil_loss=pil_loss.detach(),
            f1_acc=f1,
        )
        weight = torch.tensor(speech.size(0), device=speech.device)
        loss, stats, weight = self._force_gatherable(loss, stats, weight)
        return loss, stats, weight

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _align_targets(self, preds, preds_lengths, spk_labels, spk_labels_lengths):
        # Match the target time axis to the prediction time axis: pad with zeros
        # when labels are shorter, truncate when longer, and cap the per-item
        # valid length to the shorter of prediction/label lengths.
        t_pred = preds.size(1)
        t_lab = spk_labels.size(1)
        spk_labels = spk_labels.to(preds.dtype)
        if t_lab < t_pred:
            pad = spk_labels.new_zeros(
                spk_labels.size(0), t_pred - t_lab, spk_labels.size(2)
            )
            targets = torch.cat([spk_labels, pad], dim=1)
        else:
            targets = spk_labels[:, :t_pred, :]
        if spk_labels_lengths is None:
            spk_labels_lengths = preds_lengths
        target_lens = torch.minimum(preds_lengths, spk_labels_lengths).clamp(max=t_pred)
        return targets, target_lens

    @staticmethod
    def _frame_f1(preds, targets, target_lens, thres: float = 0.5):
        """Permutation-agnostic frame-level F1 against the (already sorted) targets."""
        pred_bin = (preds > thres).float()
        mask = torch.arange(preds.size(1), device=preds.device).expand(
            preds.size(0), preds.size(1)
        ) < target_lens.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        tp = (pred_bin * targets * mask).sum()
        fp = (pred_bin * (1 - targets) * mask).sum()
        fn = ((1 - pred_bin) * targets * mask).sum()
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom > 0 else torch.tensor(0.0, device=preds.device)

    @staticmethod
    def _force_gatherable(loss, stats, weight):
        from espnet2.torch_utils.device_funcs import force_gatherable

        return force_gatherable((loss, stats, weight), loss.device)

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return log-mel features for the ESPnet feature-collection stage.

        Args:
            speech: Waveform ``(B, n_samples)``.
            speech_lengths: Valid samples per item; defaults to the full length.
            **kwargs: Ignored extra batch fields.

        Returns:
            Dict with ``feats`` ``(B, T_feat, n_mels)`` and ``feats_lengths``.
        """
        if speech_lengths is None:
            speech_lengths = speech.new_full(
                (speech.size(0),), speech.size(1), dtype=torch.long
            )
        feats, feat_lengths = self.preprocessor(speech, speech_lengths)
        return {"feats": feats.transpose(1, 2), "feats_lengths": feat_lengths}

    @torch.no_grad()
    def diarize(
        self, speech: torch.Tensor, speech_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Offline inference entry point.

        Sets eval mode and runs the offline full-context :meth:`encode` under
        ``no_grad``. For long-form audio use :meth:`diarize_streaming` instead.

        Args:
            speech: Waveform ``(B, n_samples)``.
            speech_lengths: Valid samples per item; defaults to the full length.

        Returns:
            Tuple of speaker probabilities ``(B, T, num_spk)`` and output
            lengths.
        """
        if speech_lengths is None:
            speech_lengths = speech.new_full(
                (speech.size(0),), speech.size(1), dtype=torch.long
            )
        self.eval()
        return self.encode(speech, speech_lengths)


def build_sortformer_model(
    num_spk: int = 4,
    fc_d_model: int = 512,
    fc_n_layers: int = 18,
    fc_n_heads: int = 8,
    fc_ff_expansion_factor: int = 4,
    fc_conv_kernel_size: int = 9,
    subsampling_factor: int = 8,
    subsampling_conv_channels: int = 256,
    fc_dropout: float = 0.1,
    fc_dropout_att: float = 0.1,
    tf_d_model: int = 192,
    tf_n_layers: int = 18,
    tf_n_heads: int = 8,
    tf_inner_size: int = 768,
    tf_dropout: float = 0.5,
    sortformer_dropout: float = 0.5,
    ats_weight: float = 0.5,
    pil_weight: float = 0.5,
    n_mels: int = 80,
    sample_rate: int = 16000,
) -> ESPnetSortformerModel:
    """Build an :class:`ESPnetSortformerModel` with the released architecture.

    The default arguments reproduce ``nvidia/diar_sortformer_4spk-v1`` exactly,
    so the converted checkpoint loads with a near-identity key remap. The same
    instance supports both offline (:meth:`ESPnetSortformerModel.diarize`) and
    streaming (:meth:`ESPnetSortformerModel.diarize_streaming`) inference;
    streaming hyper-parameters live on :class:`SortformerModules`.

    Args:
        num_spk: Number of output speaker channels.
        fc_d_model: FastConformer model dimension.
        fc_n_layers: Number of FastConformer layers.
        fc_n_heads: FastConformer attention heads.
        fc_ff_expansion_factor: Feed-forward expansion factor in FastConformer.
        fc_conv_kernel_size: FastConformer depthwise-conv kernel size.
        subsampling_factor: Convolutional sub-sampling factor (8 -> ~80 ms
            frames).
        subsampling_conv_channels: Channels in the sub-sampling convolutions.
        fc_dropout: FastConformer dropout.
        fc_dropout_att: FastConformer attention dropout.
        tf_d_model: Transformer (and ``encoder_proj`` output) dimension.
        tf_n_layers: Number of Transformer layers.
        tf_n_heads: Transformer attention heads.
        tf_inner_size: Transformer feed-forward inner size.
        tf_dropout: Transformer dropout (attention and feed-forward).
        sortformer_dropout: Dropout in the sigmoid speaker head.
        ats_weight: Weight of the Arrival-Time-Sort loss term.
        pil_weight: Weight of the Permutation-Invariant loss term.
        n_mels: Number of log-mel features.
        sample_rate: Input sample rate in Hz.

    Returns:
        A configured :class:`ESPnetSortformerModel`.

    Example:
        >>> model = build_sortformer_model(num_spk=4)
        >>> preds, lengths = model.diarize_streaming(long_speech)
    """
    preprocessor = MelSpectrogramPreprocessor(sample_rate=sample_rate, features=n_mels)
    encoder = FastConformerEncoder(
        feat_in=n_mels,
        d_model=fc_d_model,
        n_layers=fc_n_layers,
        n_heads=fc_n_heads,
        ff_expansion_factor=fc_ff_expansion_factor,
        subsampling_factor=subsampling_factor,
        subsampling_conv_channels=subsampling_conv_channels,
        conv_kernel_size=fc_conv_kernel_size,
        dropout=fc_dropout,
        dropout_att=fc_dropout_att,
    )
    sortformer_modules = SortformerModules(
        num_spks=num_spk,
        dropout_rate=sortformer_dropout,
        fc_d_model=fc_d_model,
        tf_d_model=tf_d_model,
    )
    transformer_encoder = TransformerEncoder(
        num_layers=tf_n_layers,
        hidden_size=tf_d_model,
        inner_size=tf_inner_size,
        num_attention_heads=tf_n_heads,
        attn_score_dropout=tf_dropout,
        attn_layer_dropout=tf_dropout,
        ffn_dropout=tf_dropout,
    )
    return ESPnetSortformerModel(
        preprocessor=preprocessor,
        encoder=encoder,
        sortformer_modules=sortformer_modules,
        transformer_encoder=transformer_encoder,
        num_spk=num_spk,
        ats_weight=ats_weight,
        pil_weight=pil_weight,
    )
