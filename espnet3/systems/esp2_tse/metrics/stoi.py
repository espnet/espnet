"""STOI and ESTOI metric utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from espnet3.contract import Field
from espnet3.systems.esp2_tse.src.audio_metrics import AudioMetric, read_audio


class STOI(AudioMetric):
    """Compute Short-Time Objective Intelligibility (STOI) score
    for reference/hypothesis speech pairs.

    This metric expects extracted speech and reference speech as input,
    and produces an objective score as output.

    Reference:
        Cees H. Taal, Richard C. Hendriks, Richard Heusdens, Jesper Jensen.
        An algorithm for intelligibility prediction of time--frequency weighted
        noisy speech.
        IEEE Transactions on Audio, Speech, and Language Processing,
        vol. 19, no. 7, pp. 2125-2136, 2011.
    """  # noqa: H405, D205, D400

    inputs = (
        Field("ref", "audio"),
        Field("hyp", "audio"),
    )
    outputs = (Field("STOI", "number"),)

    def __init__(
        self,
        batch_size: int = 1,
        ref_channel: int = 0,
    ) -> None:
        """Initialize the STOI measure.

        Args:
            batch_size: batch size for batched inference.
            ref_channel: Reference channel index for aligning multi-channel signals.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.ref_channel = ref_channel
        self.device = "cpu"
        self.name = "STOI"
        self.extended = False

        self._init_scores()

    def _ensure_stoi(self):
        try:
            from pystoi import stoi as pystoi_stoi
        except ImportError as exc:
            raise RuntimeError("'pystoi' is required to compute STOI/ESTOI") from exc
        return pystoi_stoi

    def _align_shape(self, ref, inf):
        """Align the shape of reference and inference signals."""
        if ref.ndim > inf.ndim:
            ref = ref[..., self.ref_channel]
        elif ref.ndim < inf.ndim:
            inf = inf[..., self.ref_channel]
        elif ref.ndim == inf.ndim == 2:
            ref = ref[..., self.ref_channel]
            inf = inf[..., self.ref_channel]
        elif ref.ndim == inf.ndim != 1:
            raise ValueError(
                "Reference and inference must have the same shape, "
                f"but got {ref.shape} and {inf.shape}"
            )
        # Only accepts 1D signals
        assert ref.ndim == inf.ndim == 1, (ref.shape, inf.shape)
        return ref, inf

    def _load_audio_pairs(
        self, ref_batch: List[str], inf_batch: List[str]
    ) -> Tuple[List[Tuple[int, Tensor]], List[Tuple[int, Tensor]]]:
        """Load input audio and clean reference audio as paired data."""
        ref_audios, inf_audios = [], []
        for ref_path, inf_path in zip(ref_batch, inf_batch):
            ref_audio, sr1 = read_audio(ref_path)
            inf_audio, sr2 = read_audio(inf_path)
            ref_audio = torch.as_tensor(ref_audio, device=self.device)
            inf_audio = torch.as_tensor(inf_audio, device=self.device)
            if sr1 != sr2:
                raise ValueError(f"Sampling rates must match, but got {sr1} and {sr2}")
            ref_audio, inf_audio = self._align_shape(ref_audio, inf_audio)
            if ref_audio.shape != inf_audio.shape:
                raise ValueError(
                    "Reference and inference shapes must match: "
                    f"{ref_audio.shape}, {inf_audio.shape}"
                )
            ref_audios.append((sr1, ref_audio))
            inf_audios.append((sr2, inf_audio))
        return ref_audios, inf_audios

    def _compute_stoi(
        self, ref_audio: Tensor, inf_audio: Tensor, sample_rate: int
    ) -> Dict[str, float]:
        """Compute STOI score for a single reference/extracted audio pair."""
        stoi_fn = self._ensure_stoi()
        score = stoi_fn(
            ref_audio.cpu().numpy(),
            inf_audio.cpu().numpy(),
            fs_sig=sample_rate,
            extended=self.extended,
        )
        return float(score * 100.0)

    def update(self, *, ref, hyp):
        """Accumulate per-utterance STOI scores."""
        refs, hyps = self._load_audio_pairs(ref, hyp)
        for (rate, reference), (_, estimate) in zip(refs, hyps):
            score = self._compute_stoi(reference, estimate, rate)
            self._add_scores({self.name: score}, skip_nan=False)


class ESTOI(STOI):
    """Compute Extended Short-Time Objective Intelligibility (ESTOI) score
    for reference/hypothesis speech pairs.

    This metric expects extracted speech and reference speech as input,
    and produces an objective score as output.

    Reference:
        Jesper Jensen, Cees H Taal.
        An algorithm for predicting the intelligibility of speech masked by
        modulated noise maskers.
        IEEE/ACM Transactions on Audio, Speech, and Language Processing,
        vol. 24, no. 11, pp. 2009-2022, 2016.
    """  # noqa: H405, D205, D400

    outputs = (Field("ESTOI", "number"),)

    def __init__(
        self,
        batch_size: int = 1,
        ref_channel: int = 0,
    ) -> None:
        """Initialize the ESTOI measure.

        Args:
            batch_size: batch size for batched inference.
            ref_channel: Reference channel index for aligning multi-channel signals.
        """
        super().__init__(
            batch_size=batch_size,
            ref_channel=ref_channel,
        )
        self.name = "ESTOI"
        self.extended = True
