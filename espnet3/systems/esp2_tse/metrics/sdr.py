"""SDR/SAR/SIR metric utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from espnet3.contract import Field
from espnet3.systems.esp2_tse.src.audio_metrics import AudioMetric, read_audio


class SDR(AudioMetric):
    """Compute Signal-To-Distortion Ratio (SDR), Signal-to-Artifact Ratio (SAR),
    and Signal-to-Interference Ratio (SIR) scores for reference/hypothesis speech pairs.

    This metric expects extracted speech and reference speech as input,
    and produces objective scores (SDR, SIR, SAR) as output.

    Reference:
        Emmanuel Vincent, Rémi Gribonval, Cédric Févotte.
        Performance measurement in blind audio source separation.
        IEEE Transactions on Audio, Speech, and Language Processing,
        vol. 14, no. 4, pp. 1462-1469, 2006.
    """  # noqa: H405, D205, D400

    inputs = (
        Field("ref", "audio"),
        Field("hyp", "audio"),
    )
    outputs = (
        Field("SDR", "number"),
        Field("SAR", "number"),
        Field("SIR", "number"),
    )

    def __init__(
        self,
        batch_size: int = 1,
        ref_channel: int = 0,
        device: str = "cpu",
        # --- Below are args for bss_eval_sources ---
        filter_length: Optional[int] = 512,
        use_cg_iter: Optional[int] = None,
        zero_mean: Optional[bool] = False,
        clamp_db: Optional[float] = None,
        load_diag: Optional[float] = None,
    ) -> None:
        """Initialize the SDR, SAR, and SIR measures.

        Args:
            batch_size: batch size for batched inference.
            ref_channel: Reference channel index for aligning multi-channel signals.
            device: device to use for inference.
            filter_length: Filter length for bss_eval_sources.
            use_cg_iter: Number of conjugate gradient iterations for bss_eval_sources.
            zero_mean: Whether to zero-mean the signals before computing SDR.
            clamp_db: Optional clamping value for SDR computation.
            load_diag: Optional load_diag parameter for bss_eval_sources.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.ref_channel = ref_channel
        self.device = device

        self.filter_length = filter_length
        self.use_cg_iter = use_cg_iter
        self.zero_mean = zero_mean
        self.clamp_db = clamp_db
        self.load_diag = load_diag

        self._init_scores()

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

    def _compute_sdr_sar_sir(
        self, ref_audio: Tensor, inf_audio: Tensor
    ) -> Tuple[float, float, float]:
        """Compute SDR, SAR, and SIR scores for a reference/extracted audio pair."""
        from fast_bss_eval import bss_eval_sources

        sdr, sir, sar = bss_eval_sources(
            ref_audio.unsqueeze(0),
            inf_audio.unsqueeze(0),
            filter_length=self.filter_length,
            use_cg_iter=self.use_cg_iter,
            zero_mean=self.zero_mean,
            clamp_db=self.clamp_db,
            compute_permutation=False,
            load_diag=self.load_diag,
        )
        return float(sdr[0]), float(sar[0]), float(sir[0])

    def update(self, *, ref, hyp):
        """Accumulate per-utterance SDR, SAR and SIR."""
        refs, hyps = self._load_audio_pairs(ref, hyp)
        for (_, reference), (_, estimate) in zip(refs, hyps):
            scores = self._compute_sdr_sar_sir(reference, estimate)
            self._add_scores(dict(zip(("SDR", "SAR", "SIR"), scores)))
