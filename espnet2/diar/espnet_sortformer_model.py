"""ESPnet wrapper for the offline Sortformer speaker-diarization model.

NeMo-free reimplementation of NVIDIA's ``SortformerEncLabelModel`` (offline) as
an :class:`~espnet2.train.abs_espnet_model.AbsESPnetModel`. The pipeline is::

    audio -> peak-normalize -> log-mel (MelSpectrogramPreprocessor)
          -> FastConformerEncoder (8x subsampling)
          -> encoder_proj (512 -> 192)
          -> TransformerEncoder (post-LN)
          -> sigmoid speaker head -> per-frame speaker probabilities (B, T, 4)

Training uses the hybrid Arrival-Time-Sort + Permutation-Invariant BCE loss
(:class:`~espnet2.diar.sortformer.sort_loss.SortformerHybridLoss`).

The component sub-modules mirror the released ``nvidia/diar_sortformer_4spk-v1``
checkpoint so the weights can be converted with a near-identity key remap.
"""

import math
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch

from espnet2.diar.sortformer.fastconformer_encoder import FastConformerEncoder
from espnet2.diar.sortformer.preprocessor import MelSpectrogramPreprocessor
from espnet2.diar.sortformer.sort_loss import SortformerHybridLoss
from espnet2.diar.sortformer.sortformer_modules import SortformerModules
from espnet2.diar.sortformer.transformer_encoder import TransformerEncoder
from espnet2.train.abs_espnet_model import AbsESPnetModel


@contextmanager
def _null_context():
    yield


class ESPnetSortformerModel(AbsESPnetModel):
    """Offline Sortformer diarization model (4 speakers by default)."""

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
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.sortformer_modules = sortformer_modules
        self.transformer_encoder = transformer_encoder
        self.num_spk = num_spk
        self.eps = eps
        self.loss = SortformerHybridLoss(
            num_spks=num_spk, ats_weight=ats_weight, pil_weight=pil_weight
        )

    # ------------------------------------------------------------------ #
    # core forward
    # ------------------------------------------------------------------ #
    def _process_signal(self, speech, speech_lengths):
        # Peak-normalize the waveform (NeMo does this in process_signal, using
        # the global max over the batch tensor, eps=1e-3).
        speech = (1.0 / (speech.max() + self.eps)) * speech
        feats, feat_lengths = self.preprocessor(speech, speech_lengths)
        feats = feats[:, :, : feat_lengths.max()]
        return feats, feat_lengths  # (B, n_mels, T_feat)

    def frontend_encoder(self, x, lengths, bypass_pre_encode: bool = False):
        """Features/embeddings -> projected encoder states ``(B, T, tf_d_model)``.

        ``x`` is ``(B, n_mels, T)`` log-mel features when ``bypass_pre_encode=False``,
        or ``(B, T', fc_d_model)`` pre-encoded embeddings when ``True``.
        """
        if not bypass_pre_encode:
            x = x.transpose(1, 2)  # (B, T_feat, n_mels)
        emb, emb_lengths, _ = self.encoder(
            x, lengths, bypass_pre_encode=bypass_pre_encode
        )
        emb = self.sortformer_modules.encoder_proj(emb)
        return emb, emb_lengths

    def forward_infer(self, emb, emb_lengths):
        trans = self.transformer_encoder(emb, emb_lengths)
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans)
        max_len = preds.size(1)
        valid = torch.arange(max_len, device=preds.device).expand(
            preds.size(0), max_len
        ) < emb_lengths.unsqueeze(1)
        return preds * valid.unsqueeze(-1)

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Offline: audio -> per-frame speaker probabilities ``(B, T, num_spk)``."""
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
        """feats: (B, n_mels, T_feat). Returns total_preds (B, T_diar, num_spk)."""
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
            concat_len = state.spkcache.shape[1] + state.fifo.shape[1] + chunk_pre_len
            emb, emb_len = self.frontend_encoder(
                concat, concat_len, bypass_pre_encode=True
            )
            preds = self.forward_infer(emb, emb_len)
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
        """Long-form streaming inference. Returns (preds (B, T, num_spk), lengths)."""
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
        if speech_lengths is None:
            speech_lengths = speech.new_full(
                (speech.size(0),), speech.size(1), dtype=torch.long
            )
        assert spk_labels is not None, "spk_labels are required for training"

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
        """Inference: returns ``(preds (B, T, num_spk), preds_lengths)``."""
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
    """Construct an offline Sortformer model with NVIDIA's released architecture.

    Defaults reproduce ``nvidia/diar_sortformer_4spk-v1`` exactly so that the
    converted checkpoint loads cleanly.
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
