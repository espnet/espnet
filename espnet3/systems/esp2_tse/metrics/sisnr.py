"""Scale-invariant signal-to-noise ratio (SI-SNR) utilities."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from espnet3.contract import Field
from espnet3.systems.esp2_tse.src.audio_metrics import AudioMetric, read_audio


class SISNR(AudioMetric):
    """Compute the scale-invariant signal-to-noise ratio (SI-SNR) for speech pairs.

    This metric expects extracted speech and reference speech as input,
    and produces an objective score as output.

    Reference:
        [1] Yusuf Isik, Jonathan Le Roux, Zhuo Chen, Shinji Watanabe, John R. Hershey.
            Single-channel multi-speaker separation using deep clustering.
            in Proc. ISCA Interspeech, 2016, pp. 545-549.
        [2] Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, John R Hershey.
            SDR--half-baked or well done?.
            in Proc. IEEE ICASSP, 2019, pp. 626-630.
    """  # noqa: H405, D205, D400

    inputs = (
        Field("ref", "audio"),
        Field("hyp", "audio"),
    )
    outputs = (Field("SI_SNR", "number"),)

    def __init__(
        self,
        batch_size: int = 1,
        ref_channel: int = 0,
        device: str = "cpu",
        loss=None,
        clamp_db: float = 100.0,
        zero_mean: bool = True,
    ) -> None:
        """Initialize the SI-SNR measure.

        Args:
            batch_size: batch size for batched inference.
            ref_channel: Reference channel index for aligning multi-channel signals.
            device: device to use for inference.
            loss: Optional loss function for computing SI-SNR.
            clamp_db: Optional clamping value for SI-SNR computation.
            zero_mean: Optional flag to zero-mean the signals before computing SI-SNR.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.ref_channel = ref_channel
        self.device = device
        self.loss = loss
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean

        self._init_scores()

    def _ensure_loss(self):
        """Ensure that the SISNR loss function is available for use."""
        if self.loss is None:
            from espnet2.enh.loss.criterions.time_domain import SISNRLoss

            self.loss = SISNRLoss(clamp_db=self.clamp_db, zero_mean=self.zero_mean)
        return self.loss

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
        assert ref.ndim == inf.ndim <= 2, (ref.shape, inf.shape)
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
        return torch.nn.functional.pad(x, [0] * 2 * dim + list(pad), **kwargs)

    def _collate_fn(self, audios: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Collate a list of audio tensors into a batch tensor with padding."""
        assert len(audios) >= 1
        assert all(x.ndim == audios[0].ndim for x in audios)
        ilens = audios[0].new_tensor([x.size(0) for x in audios], dtype=torch.long)
        max_len = ilens.max().item()
        audios = [self._pad(x, (0, max_len - x.size(0)), dim=0) for x in audios]
        return torch.stack(audios), ilens

    def _compute_sisnr(self, ref_audio: Tensor, inf_audio: Tensor) -> float:
        """Compute SI-SNR for a single reference/extracted audio pair."""
        loss_fn = self._ensure_loss()
        if ref_audio.ndim == 1:
            ref_audio = ref_audio[None, :]
            inf_audio = inf_audio[None, :]
        elif ref_audio.ndim == 2:
            ref_audio = ref_audio.T
            inf_audio = inf_audio.T
        with torch.no_grad():
            score = -float(loss_fn(ref_audio, inf_audio).mean())
        return score

    def update(self, *, ref, hyp):
        """Accumulate per-utterance SI-SNR scores."""
        refs, hyps = self._load_audio_pairs(ref, hyp)
        for reference, estimate in zip(refs, hyps):
            self._add_scores({"SI_SNR": self._compute_sisnr(reference, estimate)})
