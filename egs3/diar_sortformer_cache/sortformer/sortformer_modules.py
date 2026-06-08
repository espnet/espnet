"""NeMo-free port of ``SortformerModules`` (offline head + streaming cache).

Includes the offline sigmoid head and the **streaming speaker cache** machinery
that maintains globally-consistent speaker identities across a long session:
a compressed cache of the most informative past *pre-encode* frames (plus an
optional FIFO queue) is re-fed through the conformer every chunk, so output
channel ``k`` stays the same speaker over the whole recording. This is the
mechanism that fixes long-form (session-level) speaker confusion.

Reference (Apache-2.0): NVIDIA/NeMo
    nemo/collections/asr/modules/sortformer_modules.py
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StreamingSortformerState:
    """Mutable streaming state carried across chunks.

    Bundles the speaker cache, the FIFO queue and the running silence profile
    that :class:`SortformerModules` updates each chunk. Created by
    :meth:`SortformerModules.init_streaming_state` and advanced by
    :meth:`SortformerModules.streaming_update`.

    Attributes:
        spkcache: Compressed cache of past pre-encode frames
            ``(B, spkcache_len, fc_d_model)``; the global speaker memory.
        spkcache_lengths: Valid cache length per item.
        spkcache_preds: Speaker probabilities for the cached frames
            ``(B, spkcache_len, n_spk)``.
        fifo: Pending pre-encode frames not yet committed to the cache
            ``(B, fifo_len, fc_d_model)``.
        fifo_lengths: Valid FIFO length per item.
        fifo_preds: Speaker probabilities for the FIFO frames.
        spk_perm: Speaker permutation applied during cache compression at
            training time (used to invert predictions), or None.
        mean_sil_emb: Running mean silence embedding ``(B, fc_d_model)`` used to
            fill disabled cache slots.
        n_sil_frames: Count of silence frames accumulated into ``mean_sil_emb``.
    """

    spkcache: Optional[torch.Tensor] = None  # (B, spkcache_len, fc_d_model)
    spkcache_lengths: Optional[torch.Tensor] = None
    spkcache_preds: Optional[torch.Tensor] = None  # (B, spkcache_len, n_spk)
    fifo: Optional[torch.Tensor] = None  # (B, fifo_len, fc_d_model)
    fifo_lengths: Optional[torch.Tensor] = None
    fifo_preds: Optional[torch.Tensor] = None
    spk_perm: Optional[torch.Tensor] = None
    mean_sil_emb: Optional[torch.Tensor] = None
    n_sil_frames: Optional[torch.Tensor] = None

    def to(self, device):
        for f in (
            "spkcache",
            "spkcache_lengths",
            "spkcache_preds",
            "fifo",
            "fifo_lengths",
            "fifo_preds",
            "spk_perm",
            "mean_sil_emb",
            "n_sil_frames",
        ):
            v = getattr(self, f)
            if v is not None:
                setattr(self, f, v.to(device))
        return self


class SortformerModules(nn.Module):
    """Offline sigmoid head plus the streaming speaker-cache machinery.

    Holds the projection from FastConformer width to Transformer width
    (``encoder_proj``), the sigmoid speaker head
    (:meth:`forward_speaker_sigmoids`), and the streaming cache used for
    long-form diarization. The cache keeps a compressed set of the most
    informative past pre-encode frames (plus an optional FIFO queue) and re-feeds
    them through the encoder every chunk, so output channel ``k`` stays bound to
    the same speaker across the whole session. This is what fixes session-level
    speaker confusion in long recordings.

    The streaming hyper-parameters mirror the released ``modules_config`` of
    ``nvidia/diar_sortformer_4spk-v1``. With the defaults each chunk and the
    cache hold ``188`` pre-encode frames (~15 s at 8x sub-sampling), the FIFO is
    empty (``fifo_len=0``, frames go straight to the cache), the cache is updated
    every ``188`` frames, and ``3`` silence slots per speaker are reserved.

    Args:
        num_spks: Number of speaker output channels.
        dropout_rate: Dropout in the sigmoid speaker head.
        fc_d_model: FastConformer (pre-encode/cache) feature width.
        tf_d_model: Transformer feature width (``encoder_proj`` output).
        subsampling_factor: Encoder sub-sampling factor relating feature frames
            to diarization frames.
        spkcache_len: Maximum number of frames kept in the speaker cache.
        fifo_len: FIFO queue capacity; 0 commits chunk frames to the cache
            immediately.
        chunk_len: Number of pre-encode frames per streaming chunk.
        spkcache_update_period: Frames popped from the FIFO into the cache per
            update.
        chunk_left_context: Left-context chunks added when slicing features.
        chunk_right_context: Right-context chunks added when slicing features.
        spkcache_sil_frames_per_spk: Silence slots reserved per speaker in the
            compressed cache.
        pred_score_threshold: Floor used when taking logs of predictions.
        sil_threshold: Total-activity threshold below which a frame is silence.
        strong_boost_rate: Fraction of per-speaker cache slots given a strong
            score boost.
        weak_boost_rate: Fraction given a weak score boost.
        min_pos_scores_rate: Fraction setting the minimum positive scores per
            speaker before low scores are disabled.
        scores_boost_latest: Additive boost for the most recent (non-cache)
            frames during compression.
        scores_add_rnd: Magnitude of random score noise added during training
            for cache-selection diversity.
        max_index: Sentinel index marking disabled (silence-filled) cache slots.
    """

    def __init__(
        self,
        num_spks: int = 4,
        dropout_rate: float = 0.5,
        fc_d_model: int = 512,
        tf_d_model: int = 192,
        # streaming hyper-parameters (from the released modules_config)
        subsampling_factor: int = 8,
        spkcache_len: int = 188,
        fifo_len: int = 0,
        chunk_len: int = 188,
        spkcache_update_period: int = 188,
        chunk_left_context: int = 1,
        chunk_right_context: int = 1,
        spkcache_sil_frames_per_spk: int = 5,
        pred_score_threshold: float = 1e-6,
        sil_threshold: float = 0.1,
        strong_boost_rate: float = 0.3,
        weak_boost_rate: float = 0.7,
        min_pos_scores_rate: float = 0.5,
        scores_boost_latest: float = 0.5,
        scores_add_rnd: float = 2.0,
        max_index: int = 10000,
    ):
        super().__init__()
        self.n_spk = num_spks
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.hidden_size = tf_d_model
        self.encoder_proj = nn.Linear(fc_d_model, tf_d_model)
        self.first_hidden_to_hidden = nn.Linear(tf_d_model, tf_d_model)
        self.single_hidden_to_spks = nn.Linear(tf_d_model, num_spks)
        # Present in the checkpoint (streaming concat path); unused here.
        self.hidden_to_spks = nn.Linear(2 * tf_d_model, num_spks)
        self.dropout = nn.Dropout(dropout_rate)

        # streaming params
        self.subsampling_factor = subsampling_factor
        self.spkcache_len = spkcache_len
        self.fifo_len = fifo_len
        self.chunk_len = chunk_len
        self.spkcache_update_period = spkcache_update_period
        self.chunk_left_context = chunk_left_context
        self.chunk_right_context = chunk_right_context
        self.spkcache_sil_frames_per_spk = spkcache_sil_frames_per_spk
        self.pred_score_threshold = pred_score_threshold
        self.sil_threshold = sil_threshold
        self.strong_boost_rate = strong_boost_rate
        self.weak_boost_rate = weak_boost_rate
        self.min_pos_scores_rate = min_pos_scores_rate
        self.scores_boost_latest = scores_boost_latest
        self.scores_add_rnd = scores_add_rnd
        self.max_index = max_index

    # ------------------------------------------------------------------ #
    # offline head
    # ------------------------------------------------------------------ #
    def forward_speaker_sigmoids(self, hidden_out: torch.Tensor) -> torch.Tensor:
        """Project Transformer states to per-frame speaker probabilities.

        Applies ReLU + dropout, a hidden linear, another ReLU + dropout, the
        speaker linear, and a final sigmoid. This is the offline diarization
        head shared by both the offline and streaming forward paths.

        Args:
            hidden_out: Transformer output ``(B, T, tf_d_model)``.

        Returns:
            Speaker probabilities ``(B, T, n_spk)`` in ``[0, 1]``.
        """
        hidden_out = self.dropout(F.relu(hidden_out))
        hidden_out = self.first_hidden_to_hidden(hidden_out)
        hidden_out = self.dropout(F.relu(hidden_out))
        spk_preds = self.single_hidden_to_spks(hidden_out)
        return torch.sigmoid(spk_preds)

    # ------------------------------------------------------------------ #
    # streaming helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def length_to_mask(lengths, max_length: int):
        # Build a (B, max_length) boolean validity mask from per-item lengths.
        arange = torch.arange(max_length, device=lengths.device)
        return arange.expand(lengths.shape[0], max_length) < lengths.unsqueeze(1)

    def init_streaming_state(self, batch_size: int = 1, device=None):
        """Create an empty :class:`StreamingSortformerState` for a new session.

        Call once before the chunk loop. The cache and FIFO start empty (zero
        length) and the silence profile starts at zero.

        Args:
            batch_size: Number of sessions processed in parallel.
            device: Device for the allocated tensors.

        Returns:
            A fresh :class:`StreamingSortformerState`.
        """
        st = StreamingSortformerState()
        st.spkcache = torch.zeros((batch_size, 0, self.fc_d_model), device=device)
        st.fifo = torch.zeros((batch_size, 0, self.fc_d_model), device=device)
        st.mean_sil_emb = torch.zeros((batch_size, self.fc_d_model), device=device)
        st.n_sil_frames = torch.zeros((batch_size,), dtype=torch.long, device=device)
        return st

    @staticmethod
    def concat_embs(list_of_tensors, dim: int = 1, device=None):
        # Concatenate [spkcache, fifo, chunk] (or similar) along the time axis.
        return torch.cat(list_of_tensors, dim=dim).to(device)

    def streaming_feat_loader(self, feat_seq, feat_seq_length, feat_seq_offset):
        """Iterate over fixed-size feature chunks for streaming inference.

        Splits the feature sequence into ``chunk_len``-frame chunks (in
        pre-encode units, scaled by ``subsampling_factor``), each padded with
        the configured left/right context. The returned lengths account for the
        offset and clamp to the valid portion of the chunk.

        Args:
            feat_seq: Log-mel features ``(B, n_mels, T_feat)``.
            feat_seq_length: Valid feature frames per item.
            feat_seq_offset: Per-item starting offset into the feature sequence.

        Yields:
            Tuple ``(chunk_idx, chunk_feat (B, T_chunk, n_mels), feat_lengths,
            left_off, right_off)`` where ``left_off``/``right_off`` are the
            context frame counts included on each side.
        """
        feat_len = feat_seq.shape[2]
        stt = 0
        chunk_idx = 0
        end = 0
        while end < feat_len:
            left = min(self.chunk_left_context * self.subsampling_factor, stt)
            end = min(stt + self.chunk_len * self.subsampling_factor, feat_len)
            right = min(
                self.chunk_right_context * self.subsampling_factor, feat_len - end
            )
            chunk = feat_seq[:, :, stt - left : end + right]
            flen = (feat_seq_length + feat_seq_offset - stt + left).clamp(
                0, chunk.shape[2]
            )
            flen = flen * (feat_seq_offset < end)
            stt = end
            yield chunk_idx, chunk.transpose(1, 2), flen, left, right
            chunk_idx += 1

    @staticmethod
    def apply_mask_to_preds(preds, lengths):
        # Zero out predictions beyond each item's valid length.
        b, n, s = preds.shape
        mask = torch.arange(n, device=preds.device).view(1, -1, 1).expand(b, -1, s)
        mask = mask < lengths.view(-1, 1, 1)
        return torch.where(mask, preds, torch.tensor(0.0, device=preds.device))

    def _get_silence_profile(self, mean_sil_emb, n_sil_frames, emb_seq, preds):
        """Update the running mean silence embedding from popped frames.

        Frames whose total speaker activity is below ``sil_threshold`` count as
        silence and are folded into the running mean. Returns the inputs
        unchanged if the batch has no silence frames.
        """
        is_sil = preds.sum(dim=2) < self.sil_threshold
        sil_count = is_sil.sum(dim=1)
        if not (sil_count > 0).any():
            return mean_sil_emb, n_sil_frames
        sil_emb_sum = torch.sum(emb_seq * is_sil.unsqueeze(-1), dim=1)
        upd_n = n_sil_frames + sil_count
        total = mean_sil_emb * n_sil_frames.unsqueeze(1) + sil_emb_sum
        return total / torch.clamp(upd_n.unsqueeze(1), min=1), upd_n

    def _get_log_pred_scores(self, preds):
        # Per-frame log-likelihood-ratio scores used to rank cache candidates;
        # higher means a more confident, informative speaker frame.
        lp = torch.log(torch.clamp(preds, min=self.pred_score_threshold))
        l1 = torch.log(torch.clamp(1.0 - preds, min=self.pred_score_threshold))
        l1sum = l1.sum(dim=2).unsqueeze(-1).expand(-1, -1, self.n_spk)
        return lp - l1 + l1sum - math.log(0.5)

    def _disable_low_scores(self, preds, scores, min_pos_scores_per_spk: int):
        """Mask out weak cache candidates by setting their score to ``-inf``.

        Non-speech frames are always disabled. Non-positive-score speech frames
        are disabled only when the speaker already has enough positive frames, so
        a speaker is never starved of cache slots.
        """
        is_speech = preds > 0.5
        scores = torch.where(
            is_speech, scores, torch.tensor(float("-inf"), device=scores.device)
        )
        is_pos = scores > 0
        repl = (
            (~is_pos)
            * is_speech
            * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores_per_spk)
        )
        scores = torch.where(
            repl, torch.tensor(float("-inf"), device=scores.device), scores
        )
        return scores

    def _get_max_perm_index(self, scores):
        # Highest speaker channel index that is still safe to permute per item:
        # the first channel with no positive scores (or n_spk if all are active).
        batch_size, _, n_spk = scores.shape
        is_pos = scores > 0
        zero_idx = torch.where(is_pos.sum(dim=1) == 0)
        max_perm = torch.full(
            (batch_size,), n_spk, dtype=torch.long, device=scores.device
        )
        max_perm.scatter_reduce_(
            0, zero_idx[0], zero_idx[1], reduce="amin", include_self=False
        )
        return max_perm

    def _permute_speakers(self, scores, max_perm_index):
        # Randomly permute the active speaker channels (training augmentation);
        # also return the permutation so predictions can be inverted later.
        spk_perm_list, scores_list = [], []
        batch_size, _, n_spk = scores.shape
        for b in range(batch_size):
            rp = torch.randperm(int(max_perm_index[b].item()))
            lin = torch.arange(int(max_perm_index[b].item()), n_spk)
            perm = torch.cat([rp, lin])
            spk_perm_list.append(perm)
            scores_list.append(scores[b, :, perm])
        return (
            torch.stack(scores_list).to(scores.device),
            torch.stack(spk_perm_list).to(scores.device),
        )

    def _boost_topk_scores(
        self,
        scores,
        n_boost_per_spk: int,
        scale_factor: float = 1.0,
        offset: float = 0.5,
    ):
        # Boost the top-k scoring frames per speaker so they survive selection;
        # keeps the most confident frames of each speaker in the cache.
        batch_size, _, n_spk = scores.shape
        if n_boost_per_spk <= 0:
            return scores
        _, topk = torch.topk(scores, n_boost_per_spk, dim=1, largest=True, sorted=False)
        bi = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
        si = torch.arange(n_spk).unsqueeze(0).unsqueeze(0)
        scores[bi, topk, si] -= scale_factor * math.log(offset)
        return scores

    def _get_topk_indices(self, scores):
        """Pick the ``spkcache_len`` frames to keep, sorted by original time.

        Selects the top scoring frame/speaker entries, marks ``-inf`` (disabled)
        picks and reserved silence-pad slots, and returns the chosen frame
        indices in time order plus a boolean ``is_disabled`` mask for slots that
        must be filled with the silence embedding instead.
        """
        batch_size, n_frames, _ = scores.shape
        n_frames_no_sil = n_frames - self.spkcache_sil_frames_per_spk
        flat = scores.permute(0, 2, 1).reshape(batch_size, -1)
        topk_values, topk_indices = torch.topk(
            flat, self.spkcache_len, dim=1, sorted=False
        )
        valid = topk_values != float("-inf")
        topk_indices = torch.where(
            valid, topk_indices, torch.tensor(self.max_index, device=scores.device)
        )
        topk_sorted, _ = torch.sort(topk_indices, dim=1)
        is_disabled = topk_sorted == self.max_index
        topk_sorted = torch.remainder(topk_sorted, n_frames)
        is_disabled = is_disabled + (topk_sorted >= n_frames_no_sil)
        topk_sorted[is_disabled] = 0
        return topk_sorted, is_disabled

    def _gather_spkcache_and_preds(
        self, emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
    ):
        # Gather the selected embeddings and predictions; disabled slots are
        # filled with the mean silence embedding and zero predictions.
        emb_dim, n_spk = emb_seq.shape[2], preds.shape[2]
        idx_emb = topk_indices.unsqueeze(-1).expand(-1, -1, emb_dim)
        emb_g = torch.gather(emb_seq, 1, idx_emb)
        sil = mean_sil_emb.unsqueeze(1).expand(-1, self.spkcache_len, -1)
        emb_g = torch.where(is_disabled.unsqueeze(-1), sil, emb_g)
        idx_spk = topk_indices.unsqueeze(-1).expand(-1, -1, n_spk)
        preds_g = torch.gather(preds, 1, idx_spk)
        preds_g = torch.where(
            is_disabled.unsqueeze(-1), torch.tensor(0.0, device=preds.device), preds_g
        )
        return emb_g, preds_g

    def _compress_spkcache(
        self, emb_seq, preds, mean_sil_emb, permute_spk: bool = False
    ):
        """Compress an over-long cache back down to ``spkcache_len`` frames.

        Scores every candidate frame, disables weak ones, optionally permutes
        speakers and adds random noise (training only), boosts each speaker's top
        frames, reserves silence slots, then selects and gathers the surviving
        frames. Returns ``(spkcache, spkcache_preds, spk_perm)`` where
        ``spk_perm`` is the applied speaker permutation (or None).
        """
        batch_size, n_frames, n_spk = preds.shape
        per_spk = self.spkcache_len // n_spk - self.spkcache_sil_frames_per_spk
        strong = math.floor(per_spk * self.strong_boost_rate)
        weak = math.floor(per_spk * self.weak_boost_rate)
        min_pos = math.floor(per_spk * self.min_pos_scores_rate)

        scores = self._get_log_pred_scores(preds)
        scores = self._disable_low_scores(preds, scores, min_pos)
        if permute_spk:
            max_perm = self._get_max_perm_index(scores)
            scores, spk_perm = self._permute_speakers(scores, max_perm)
        else:
            spk_perm = None
        if self.scores_boost_latest > 0:
            scores[:, self.spkcache_len :, :] += self.scores_boost_latest
        if self.training and self.scores_add_rnd > 0:
            scores = (
                scores
                + torch.rand(batch_size, n_frames, n_spk, device=scores.device)
                * self.scores_add_rnd
            )
        scores = self._boost_topk_scores(scores, strong, scale_factor=2)
        scores = self._boost_topk_scores(scores, weak, scale_factor=1)
        if self.spkcache_sil_frames_per_spk > 0:
            pad = torch.full(
                (batch_size, self.spkcache_sil_frames_per_spk, n_spk),
                float("inf"),
                device=scores.device,
            )
            scores = torch.cat([scores, pad], dim=1)
        topk_indices, is_disabled = self._get_topk_indices(scores)
        spkcache, spkcache_preds = self._gather_spkcache_and_preds(
            emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
        )
        return spkcache, spkcache_preds, spk_perm

    def streaming_update(self, streaming_state, chunk, preds, lc: int = 0, rc: int = 0):
        """Advance the cache/FIFO with one chunk and return its predictions.

        Called once per chunk after the encoder/head have produced predictions
        over the concatenated ``[spkcache, fifo, chunk]`` sequence. The steps
        are: invert any training-time speaker permutation; slice the chunk's own
        frames (dropping left/right context) and their predictions; push the
        chunk into the FIFO; once the FIFO overflows, pop frames into the speaker
        cache (updating the silence profile) and compress the cache back to
        ``spkcache_len`` if it grew too long.

        Args:
            streaming_state: The :class:`StreamingSortformerState` to advance
                (mutated in place and also returned).
            chunk: Pre-encode embeddings for this chunk including context
                ``(B, lc + chunk_len + rc, fc_d_model)``.
            preds: Speaker probabilities over ``[spkcache, fifo, chunk]``
                ``(B, T_concat, n_spk)``.
            lc: Left-context frames to drop from the chunk.
            rc: Right-context frames to drop from the chunk.

        Returns:
            Tuple of the updated state and the chunk's own predictions
            ``(B, chunk_len, n_spk)``.
        """
        batch_size = chunk.shape[0]
        spkcache_len = streaming_state.spkcache.shape[1]
        fifo_len = streaming_state.fifo.shape[1]
        chunk_len = chunk.shape[1] - lc - rc

        if streaming_state.spk_perm is not None:
            inv = torch.stack(
                [torch.argsort(streaming_state.spk_perm[b]) for b in range(batch_size)]
            )
            preds = torch.stack([preds[b, :, inv[b]] for b in range(batch_size)])

        streaming_state.fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
        chunk = chunk[:, lc : chunk_len + lc]
        chunk_preds = preds[
            :, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + chunk_len + lc
        ]

        streaming_state.fifo = torch.cat([streaming_state.fifo, chunk], dim=1)
        streaming_state.fifo_preds = torch.cat(
            [streaming_state.fifo_preds, chunk_preds], dim=1
        )

        if fifo_len + chunk_len > self.fifo_len:
            pop = self.spkcache_update_period
            pop = max(pop, chunk_len - self.fifo_len + fifo_len)
            pop = min(pop, fifo_len + chunk_len)
            pop_embs = streaming_state.fifo[:, :pop]
            pop_preds = streaming_state.fifo_preds[:, :pop]
            streaming_state.mean_sil_emb, streaming_state.n_sil_frames = (
                self._get_silence_profile(
                    streaming_state.mean_sil_emb,
                    streaming_state.n_sil_frames,
                    pop_embs,
                    pop_preds,
                )
            )
            streaming_state.fifo = streaming_state.fifo[:, pop:]
            streaming_state.fifo_preds = streaming_state.fifo_preds[:, pop:]
            streaming_state.spkcache = torch.cat(
                [streaming_state.spkcache, pop_embs], dim=1
            )
            if streaming_state.spkcache_preds is not None:
                streaming_state.spkcache_preds = torch.cat(
                    [streaming_state.spkcache_preds, pop_preds], dim=1
                )
            if streaming_state.spkcache.shape[1] > self.spkcache_len:
                if streaming_state.spkcache_preds is None:
                    streaming_state.spkcache_preds = torch.cat(
                        [preds[:, :spkcache_len], pop_preds], dim=1
                    )
                (
                    streaming_state.spkcache,
                    streaming_state.spkcache_preds,
                    streaming_state.spk_perm,
                ) = self._compress_spkcache(
                    emb_seq=streaming_state.spkcache,
                    preds=streaming_state.spkcache_preds,
                    mean_sil_emb=streaming_state.mean_sil_emb,
                    permute_spk=self.training,
                )
        return streaming_state, chunk_preds
