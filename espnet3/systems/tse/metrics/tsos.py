"""Target speaker over-suppression (TSOS) measure utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor

from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet3.components.metrics.base_metric import BaseMetric


class TSOS(BaseMetric):
    """Compute the TSOS measure for extracted speech.

    This metric expects extracted speech and reference speech and produces a
    percentage score.

    Reference:
        Sefik Emre Eskimez, Takuya Yoshioka, Huaming Wang, Xiaofei Wang, Zhuo Chen,
        Xuedong Huang. Personalized speech enhancement: New models and comprehensive
        evaluation. in Proc. IEEE ICASSP, 2022, pp. 356-360.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "inf",
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
            ref_key: Key name for reference speech.
            hyp_key: Key name for extracted speech.
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
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        assert isinstance(batch_size, int) and batch_size > 0, batch_size
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

    def _align_shape(self, ref, inf):
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
        return ref, inf

    def load_audio_pairs(
        self, ref_batch: List[str], inf_batch: List[str]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        ref_audios, inf_audios = [], []
        for ref_path, inf_path in zip(ref_batch, inf_batch):
            ref_audio, sr1 = sf.read(ref_path, dtype="float32")
            inf_audio, sr2 = sf.read(inf_path, dtype="float32")
            ref_audio = torch.as_tensor(ref_audio, device=self.device)
            inf_audio = torch.as_tensor(inf_audio, device=self.device)
            assert sr1 == sr2, f"Sampling rates must match, but got {sr1} and {sr2}"
            ref_audio, inf_audio = self._align_shape(ref_audio, inf_audio)
            ref_audios.append(ref_audio)
            inf_audios.append(inf_audio)
        return ref_audios, inf_audios

    def _pad(self, x: Tensor, pad: Tuple[int, int], dim: int = -1, **kwargs) -> Tensor:
        dim = x.ndim - dim - 1 if dim >= 0 else -dim - 1
        return F.pad(x, [0] * 2 * dim + list(pad), **kwargs)

    def collate_fn(self, audios: List[Tensor]) -> Tuple[Tensor, Tensor]:
        assert len(audios) >= 1
        assert all(x.ndim == audios[0].ndim for x in audios)
        ilens = audios[0].new_tensor([x.size(0) for x in audios], dtype=torch.long)
        max_len = ilens.max().item()
        audios = [self._pad(x, (0, max_len - x.size(0)), dim=0) for x in audios]
        return torch.stack(audios), ilens

    def compute_tsos(
        self, ref_spec: Tensor, inf_spec: Tensor, flens: Tensor
    ) -> List[float]:
        # [Batch, Frame, (Channel,) Freq]
        ref_mag, inf_mag = ref_spec.abs() ** self.power, inf_spec.abs() ** self.power
        oversuppression = torch.clamp_min(ref_mag - inf_mag, 0.0) ** 2
        dims = tuple(i for i in range(ref_mag.ndim) if i > 1)
        tsos = oversuppression.sum(dim=dims) > ref_mag.sum(dim=dims)
        return tsos.mean(-1).cpu().tolist()

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        """Compute TSOS, and return the metric.

        Args:
            data: Mapping of field names to SCP file paths.
                Expected keys: ref_key (default "ref") and hyp_key (default "inf").
            test_name: Test set name used for output directory naming.
            inference_dir: Base directory for storing per-sample metrics.
        """
        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        # Collect aligned (uid, ref_path, inf_path) pairs via SCP streaming
        pairs = []
        for uid, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            pairs.append((uid, row[self.ref_key], row[self.hyp_key]))

        tsos_all = []
        with (test_dir / "tsos").open("w", encoding="utf-8") as f:
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                batch_uids = [b[0] for b in batch]
                ref_paths = [b[1] for b in batch]
                inf_paths = [b[2] for b in batch]
                ref_audios, inf_audios = self.load_audio_pairs(ref_paths, inf_paths)
                ref_tensor, ref_ilens = self.collate_fn(ref_audios)
                inf_tensor, _ = self.collate_fn(inf_audios)
                ref_specs, flens = self.stft(ref_tensor, ilens=ref_ilens)
                inf_specs, _ = self.stft(inf_tensor, ilens=ref_ilens)
                tsos_batch = self.compute_tsos(ref_specs, inf_specs, flens)
                tsos_all.extend(tsos_batch)
                for uid, score in zip(batch_uids, tsos_batch):
                    f.write(f"{uid} {score}\n")

        return {"TSOS": float(np.mean(tsos_all))}
