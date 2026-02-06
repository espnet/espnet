"""Target speaker over-suppression (TSOS) measure utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor

from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet3.components.metrics.abs_metric import AbsMetric


class TSOS(AbsMetric):
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
        inf_key: str = "inf",
        batch_size: int = 1,
        power: float = 2.0,
        threshold: float = 0.1,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        device: str = "cpu",
    ) -> None:
        """Initialize the TSOS measure.

        Args:
            ref_key: Key name for reference speech.
            inf_key: Key name for extracted speech.
            batch_size: batch size for batched inference.
            power: Factor for power-law compression in TSOS calculation.
            threshold: Threshold (gamma) for TSOS calculation.
            n_fft: FFT size in STFT.
            win_length: Window length in STFT.
            hop_length: Hop length in STFT.
            window: Window function in STFT.
            device: device to use for inference.
        """
        self.ref_key = ref_key
        self.inf_key = inf_key
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
        self.device = device

    def _align_shape(self, ref, inf):
        if ref.shape != inf.shape:
            if ref.ndim > inf.ndim:
                # multi-channel reference and single-channel output
                ref = ref[..., ref_channel]
            elif ref.ndim < inf.ndim:
                # single-channel reference and multi-channel output
                inf = inf[..., ref_channel]
            elif ref.ndim == inf.ndim == 2:
                # multi-channel reference and output
                ref = ref[..., ref_channel]
                inf = inf[..., ref_channel]
            else:
                raise ValueError(
                    "Reference and inference must have the same shape, "
                    f"but got {ref.shape} and {inf.shape}"
                )
        return ref, inf

    def load_audio_pairs(
        self, ref_batch: List[str], inf_batch: List[str], ref_channel: int = 0
    ) -> Tuple[List[Tensor], List[Tensor]]:
        ref_audios, inf_audios = [], []
        for ref_path, inf_path in zip(ref_batch, inf_batch):
            ref_audio, sr1 = sf.read(ref_path, dtype="float32")
            inf_audio, sr2 = sf.read(inf_path, dtype="float32")
            ref_audio = torch.as_tensor(ref_audio, device=self.device)
            inf_audio = torch.as_tensor(inf_audio, device=self.device)
            assert sr1 == sr2, f"Sampling rates must match, but got {sr1} and {sr2}"
            ref_audio, inf_audio = self._align_shape(ref_audio, inf_audio, ref_channel)
            ref_audios.append(ref_audio)
            inf_audios.append(inf_audio)
        return ref_audios, inf_audios

    def _ensure_same_ndim(self, audios):
        assert len(audios) > 1
        assert all(x.ndim == audios[0].ndim for x in audios)

    def _pad(self, x: Tensor, pad: Tuple[int, int], dim: int = -1, **kwargs) -> Tensor:
        # Pad `pad` to the specified dimension `dim` of `x`.
        dim = x.ndim - dim - 1 if dim >= 0 else -dim - 1
        return F.pad(x, [0] * 2 * dim + list(pad), **kwargs)

    def collate_fn(self, audio_paths: List[str]) -> Tensor:
        self._ensure_same_ndim(audios)
        ilens = audios[0].new_tensor([x.size(0) for x in audios], dtype=torch.long)
        max_len = ilens.max().item()
        audios = [self._pad(x, (0, max_len - x.size(0)), dim=0) for x in audios]
        return torch.stack(audios), ilens

    def get_batches(
        self, data: Dict[str, List[str]]
    ) -> Iterable[Tuple[List[Tensor], List[Tensor]]]:
        uids = data["uttid"]
        refs, infs = data[self.ref_key], data[self.inf_key]
        assert len(refs) == len(infs), (len(refs), len(infs))
        length = len(refs)
        i = 0
        while i < length:
            uids = uids[i : i + self.batch_size]
            ref_paths = refs[i : i + self.batch_size]
            inf_paths = infs[i : i + self.batch_size]
            ref_audios, inf_audios = self.load_audio_pairs(ref_paths, inf_paths)
            # [Batch, Frames, (Channels)]
            ref_audios, ref_ilens = self.collate_fn(ref_audios)
            inf_audios, inf_ilens = self.collate_fn(inf_audios)
            assert ref_ilens == inf_ilens, (ref_ilens, inf_ilens)
            yield uids, ref_audios, inf_audios, ref_ilens
            i += self.batch_size

    def compute_tsos(self, ref_spec: Tensor, inf_spec: Tensor) -> List[float]:
        # [Batch, Frame, (Channel,) Freq]
        ref_mag, inf_mag = ref_spec.abs() ** self.power, inf_spec.abs() ** self.power
        oversuppression = torch.clamp_min(ref_mag - inf_mag, 0.0) ** 2
        dims = tuple(i for i in range(ref_mag.ndim) if i > 1)
        # Frame-level TSOS measure [Batch, Frame]
        tsos = oversuppression.sum(dim=dims) > ref_mag.sum(dim=dims)
        return tsos.mean(-1).cpu().tolist()

    def __call__(
        self,
        data: Dict[str, List[str]],
        test_name: str,
        inference_dir: Path,
        ref_channel: int = 0,
    ) -> Dict[str, float]:
        """Compute TSOS, and return the metric.

        Args:
            data: Mapping of field names to a list of strings. This is built from the
                data, e.g.
                ``{"utt_id": [s1, s2, ..], "ref": [s1, s2, ..], "inf": [s1, s2, ..]}``.
            test_name: Test set name used for output directory naming.
            inference_dir: Base directory for storing per-sample metrics.
            ref_channel: Reference channel index for aligning multi-channel signals.

        Returns:
            Dict containing mean TSOS measure.
        """
        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        tsos = []
        for uids, ref_audios, inf_audios, ilens in self.get_batches(data):
            ref_specs, flens = self.stft(ref_audios, ilens=ilens)
            inf_specs, _ = self.stft(inf_audios, ilens=ilens)
            tsos_batch = self.compute_tsos(ref_specs, inf_spec)
            tsos.extend(tsos_batch)

            with (test_dir / "tsos").open("w", encoding="utf-8") as f:
                for uid, tsos_item in zip(uids, tsos_batch):
                    f.write(f"{uid} {tsos_item}\n")

        return {"TSOS": np.mean(tsos)}
