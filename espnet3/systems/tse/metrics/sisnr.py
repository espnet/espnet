"""Scale-invariant signal-to-noise ratio (SI-SNR) utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from tqdm import trange

from espnet3.components.metrics.base_metric import BaseMetric


class SISNR(BaseMetric):
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
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "inf",
        batch_size: int = 1,
        ref_channel: int = 0,
        device: str = "cpu",
        loss=None,
        clamp_db: float = 100.0,
        zero_mean: bool = True,
    ) -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        assert isinstance(batch_size, int) and batch_size > 0, batch_size
        self.batch_size = batch_size
        self.ref_channel = ref_channel
        self.device = device
        self.loss = loss
        self.clamp_db = clamp_db
        self.zero_mean = zero_mean

    def _ensure_loss(self):
        if self.loss is None:
            from espnet2.enh.loss.criterions.time_domain import SISNRLoss

            self.loss = SISNRLoss(clamp_db=self.clamp_db, zero_mean=self.zero_mean)
        return self.loss

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
        return torch.nn.functional.pad(x, [0] * 2 * dim + list(pad), **kwargs)

    def collate_fn(self, audios: List[Tensor]) -> Tuple[Tensor, Tensor]:
        assert len(audios) >= 1
        assert all(x.ndim == audios[0].ndim for x in audios)
        ilens = audios[0].new_tensor([x.size(0) for x in audios], dtype=torch.long)
        max_len = ilens.max().item()
        audios = [self._pad(x, (0, max_len - x.size(0)), dim=0) for x in audios]
        return torch.stack(audios), ilens

    def compute_sisnr(self, ref_audio: Tensor, inf_audio: Tensor) -> float:
        loss_fn = self._ensure_loss()
        with torch.no_grad():
            score = -float(loss_fn(ref_audio[None, ...], inf_audio[None, ...]))
        return score

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        """Compute SI-SNR for each utterance and return the mean score."""
        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        pairs = []
        for uid, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            pairs.append((uid, row[self.ref_key], row[self.hyp_key]))

        scores = []
        with (test_dir / "sisnr").open("w", encoding="utf-8") as f:
            for i in trange(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                batch_uids = [sample[0] for sample in batch]
                ref_paths = [sample[1] for sample in batch]
                inf_paths = [sample[2] for sample in batch]

                ref_audios, inf_audios = self.load_audio_pairs(ref_paths, inf_paths)
                for uid, ref_audio, inf_audio in zip(
                    batch_uids, ref_audios, inf_audios
                ):
                    score = self.compute_sisnr(ref_audio, inf_audio)
                    scores.append(score)
                    f.write(f"{uid} {score}\n")

        if not scores:
            raise ValueError("No scores were computed. Please check the input data.")
        return {"SI_SNR": float(np.mean(scores))}
