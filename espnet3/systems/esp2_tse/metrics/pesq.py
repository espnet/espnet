"""PESQ metric utilities."""

from __future__ import annotations

from typing import List, Tuple

import librosa
import torch
from torch import Tensor

from espnet3.contract import Field
from espnet3.systems.esp2_tse.src.audio_metrics import AudioMetric, read_audio


class PESQ(AudioMetric):
    """Compute PESQ scores for reference/hypothesis speech pairs.

    This metric expects extracted speech and reference speech as input,
    and produces an objective score as output.

    Note: Please make sure that you have the proper license to use PESQ.

    Reference:
        Antony W Rix, John G Beerends, Michael P Hollier, Andries P Hekstra.
        Perceptual evaluation of speech quality (PESQ)---a new method for
        speech quality assessment of telephone networks and codecs.
        in Proc. IEEE ICASSP, 2001, pp. 749-752.
    """  # noqa: H405, D205, D400

    inputs = (
        Field("ref", "audio"),
        Field("hyp", "audio"),
    )
    outputs = (Field("PESQ", "number"),)

    def __init__(
        self,
        batch_size: int = 1,
        ref_channel: int = 0,
    ) -> None:
        """Initialize the PESQ measure.

        Args:
            batch_size: batch size for batched inference.
            ref_channel: Reference channel index for aligning multi-channel signals.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.ref_channel = ref_channel
        self.device = "cpu"

        self._init_scores()

    def _ensure_pesq(self):
        """Ensure that the PESQ library is available for use."""
        try:
            from pesq import PesqError, pesq
        except ImportError as exc:
            raise RuntimeError("'pesq' is required to compute PESQ") from exc
        return PesqError, pesq

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

    def _compute_pesq(self, ref_audio: Tensor, inf_audio: Tensor, sample_rate: int):
        """Compute PESQ score for a single reference/extracted audio pair."""
        PesqError, pesq_fn = self._ensure_pesq()
        if sample_rate == 8000:
            mode = "nb"
        elif sample_rate == 16000:
            mode = "wb"
        elif sample_rate > 16000:
            mode = "wb"
            ref_audio = torch.as_tensor(
                librosa.resample(
                    ref_audio.cpu().numpy(), orig_sr=sample_rate, target_sr=16000
                ),
                dtype=ref_audio.dtype,
            )
            inf_audio = torch.as_tensor(
                librosa.resample(
                    inf_audio.cpu().numpy(), orig_sr=sample_rate, target_sr=16000
                ),
                dtype=inf_audio.dtype,
            )
            sample_rate = 16000
        else:
            raise ValueError(
                "sample rate must be 8000 or 16000 for PESQ evaluation, "
                f"but got {sample_rate}"
            )

        score = pesq_fn(
            sample_rate,
            ref_audio.cpu().numpy(),
            inf_audio.cpu().numpy(),
            mode=mode,
            on_error=PesqError.RETURN_VALUES,
        )
        if score < 0:
            # Errors are returned as negative values
            return float("nan")
        return float(score)

    def update(self, *, ref, hyp):
        """Accumulate per-utterance PESQ scores."""
        refs, hyps = self._load_audio_pairs(ref, hyp)
        for (rate, reference), (_, estimate) in zip(refs, hyps):
            score = self._compute_pesq(reference, estimate, rate)
            self._add_scores({"PESQ": score}, skip_nan=True)
