"""Target speaker over-suppression (TSOS) measure utilities."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet3.contract import Field
from espnet3.systems.esp2_tse.src.audio_metrics import AudioMetric, read_audio


class TSOS(AudioMetric):
    """Compute the TSOS measure for extracted speech.

    This metric expects extracted speech and reference speech as input,
    and produces a percentage score as output.

    Reference:
        Sefik Emre Eskimez, Takuya Yoshioka, Huaming Wang, Xiaofei Wang, Zhuo Chen,
        Xuedong Huang. Personalized speech enhancement: New models and comprehensive
        evaluation. in Proc. IEEE ICASSP, 2022, pp. 356-360.
    """

    inputs = (
        Field("ref", "audio"),
        Field("hyp", "audio"),
    )
    outputs = (Field("TSOS", "number"),)

    def __init__(
        self,
        batch_size: int = 1,
        power: float = 2.0,
        threshold: float = 0.1,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        ref_channel: int = 0,
        device: str = "cpu",
    ) -> None:
        """Initialize the TSOS measure.

        Args:
            batch_size: batch size for batched inference.
            power: Factor for power-law compression in TSOS calculation.
            threshold: Threshold (gamma) for TSOS calculation.
            n_fft: FFT size in STFT.
            win_length: Window length in STFT.
            hop_length: Hop length in STFT.
            window: Window function in STFT.
            ref_channel: Reference channel index for aligning multi-channel signals.
            device: device to use for inference.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        assert isinstance(power, (float, int)) and power > 0, power
        self.power = power
        assert isinstance(threshold, float) and threshold > 0, threshold
        self.threshold = threshold

        self.stft = STFTEncoder(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
        ).to(device)
        self.ref_channel = ref_channel
        self.device = device

        self._init_scores()

    def _align_shape(self, ref, inf):
        """Align the shape of reference and inference signals."""
        if ref.shape != inf.shape:
            if ref.ndim > inf.ndim:
                ref = ref[..., self.ref_channel]
            elif ref.ndim < inf.ndim:
                inf = inf[..., self.ref_channel]
            elif ref.ndim == inf.ndim == 2:
                ref = ref[..., self.ref_channel]
                inf = inf[..., self.ref_channel]
            else:
                raise ValueError(
                    "Reference and inference must have the same shape, "
                    f"but got {ref.shape} and {inf.shape}"
                )
        if ref.shape != inf.shape:
            raise ValueError(
                "Reference and inference must have the same shape after channel "
                f"selection, but got {ref.shape} and {inf.shape}"
            )
        else:
            assert ref.ndim <= 2, ref.shape
        return ref, inf

    def _load_audio_pairs(
        self, ref_batch: List[str], inf_batch: List[str]
    ) -> Tuple[List[Tensor], List[Tensor]]:
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
            ref_audios.append(ref_audio)
            inf_audios.append(inf_audio)
        return ref_audios, inf_audios

    def _pad(self, x: Tensor, pad: Tuple[int, int], dim: int = -1, **kwargs) -> Tensor:
        """Pad a tensor along a specified dimension."""
        dim = x.ndim - dim - 1 if dim >= 0 else -dim - 1
        return F.pad(x, [0] * 2 * dim + list(pad), **kwargs)

    def _collate_fn(self, audios: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Collate a list of audio tensors into a padded batch tensor and lengths."""
        assert len(audios) >= 1
        assert all(x.ndim == audios[0].ndim for x in audios)
        ilens = audios[0].new_tensor([x.size(0) for x in audios], dtype=torch.long)
        max_len = ilens.max().item()
        audios = [self._pad(x, (0, max_len - x.size(0)), dim=0) for x in audios]
        return torch.stack(audios), ilens

    def _compute_tsos(
        self, ref_spec: Tensor, inf_spec: Tensor, flens: Tensor
    ) -> List[float]:
        """Compute TSOS for a batch of reference and inference spectrograms."""
        # [Batch, Frame, (Channel,) Freq]
        ref_mag, inf_mag = ref_spec.abs() ** self.power, inf_spec.abs() ** self.power
        oversuppression = torch.clamp_min(ref_mag - inf_mag, 0.0) ** 2
        dims = tuple(i for i in range(ref_mag.ndim) if i > 1)

        # Apply the threshold before comparing oversuppressed energy against the
        # reference energy. This matches the TSOS definition and avoids counting
        # near-equal bins as oversuppression.
        ref_energy = ref_mag.sum(dim=dims)
        oversuppression_energy = oversuppression.sum(dim=dims)
        threshold_mask = oversuppression_energy > self.threshold * ref_energy

        # Ignore padded frames for each sample using the STFT lengths produced by
        # the encoder, otherwise zero-padded frames can depress the per-sample TSOS.
        valid_mask = torch.arange(ref_spec.size(1), device=ref_spec.device).unsqueeze(0)
        valid_mask = valid_mask < flens.unsqueeze(1)
        score_dtype = ref_mag.dtype
        tsos = torch.where(
            valid_mask,
            threshold_mask.to(score_dtype),
            torch.zeros_like(threshold_mask, dtype=score_dtype),
        )
        tsos = tsos.sum(-1) / valid_mask.sum(-1).clamp_min(1).to(score_dtype)
        return tsos.cpu().tolist()

    def update(self, *, ref, hyp):
        """Accumulate TSOS with padded STFT frames excluded."""
        for start in range(0, len(ref), self.batch_size):
            refs, hyps = self._load_audio_pairs(
                ref[start : start + self.batch_size],
                hyp[start : start + self.batch_size],
            )
            ref_tensor, lengths = self._collate_fn(refs)
            hyp_tensor, _ = self._collate_fn(hyps)
            ref_spec, flens = self.stft(ref_tensor, ilens=lengths)
            hyp_spec, _ = self.stft(hyp_tensor, ilens=lengths)
            for score in self._compute_tsos(ref_spec, hyp_spec, flens):
                self._add_scores({"TSOS": score})
