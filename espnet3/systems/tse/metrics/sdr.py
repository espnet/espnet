"""SDR/SAR/SIR metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fast_bss_eval import bss_eval_sources
from torch import Tensor
from tqdm import trange

from espnet3.components.metrics.base_metric import BaseMetric


class SDR(BaseMetric):
    """Compute Signal-To-Distortion Ratio (SDR), Signal-to-Artifact Ratio (SAR),
    and Signal-to-Interference Ratio (SIR) scores for reference/hypothesis speech pairs.

    This metric expects extracted speech and reference speech as input,
    and produces objective scores (SDR, SIR, SAR) as output.

    Reference:
        Emmanuel Vincent, Rémi Gribonval, Cédric Févotte.
        Performance measurement in blind audio source separation.
        IEEE Transactions on Audio, Speech, and Language Processing,
        vol. 14, no. 4, pp. 1462-1469, 2006.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "inf",
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
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        assert isinstance(batch_size, int) and batch_size > 0, batch_size
        self.batch_size = batch_size
        self.ref_channel = ref_channel
        self.device = device

        self.filter_length = filter_length
        self.use_cg_iter = use_cg_iter
        self.zero_mean = zero_mean
        self.clamp_db = clamp_db
        self.load_diag = load_diag

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
    ) -> Tuple[List[Tuple[int, Tensor]], List[Tuple[int, Tensor]]]:
        ref_audios, inf_audios = [], []
        for ref_path, inf_path in zip(ref_batch, inf_batch):
            ref_audio, sr1 = sf.read(ref_path, dtype="float32")
            inf_audio, sr2 = sf.read(inf_path, dtype="float32")
            ref_audio = torch.as_tensor(ref_audio, device=self.device)
            inf_audio = torch.as_tensor(inf_audio, device=self.device)
            assert sr1 == sr2, f"Sampling rates must match, but got {sr1} and {sr2}"
            ref_audio, inf_audio = self._align_shape(ref_audio, inf_audio)
            ref_audios.append((sr1, ref_audio))
            inf_audios.append((sr2, inf_audio))
        return ref_audios, inf_audios

    def compute_sdr_sar_sir(
        self, ref_audio: Tensor, inf_audio: Tensor
    ) -> Tuple[float, float, float]:
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

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        pairs = []
        for uid, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            pairs.append((uid, row[self.ref_key], row[self.hyp_key]))

        sdr_scores, sar_scores, sir_scores = [], [], []
        with (test_dir / "sdr").open("w", encoding="utf-8") as f:
            for i in trange(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                batch_uids = [sample[0] for sample in batch]
                ref_paths = [sample[1] for sample in batch]
                inf_paths = [sample[2] for sample in batch]

                ref_audios, inf_audios = self.load_audio_pairs(ref_paths, inf_paths)
                for uid, ref_item, inf_item in zip(batch_uids, ref_audios, inf_audios):
                    ref_audio = ref_item[1]
                    inf_audio = inf_item[1]
                    sdr, sar, sir = self.compute_sdr_sar_sir(ref_audio, inf_audio)
                    sdr_scores.append(sdr)
                    sar_scores.append(sar)
                    sir_scores.append(sir)
                    f.write(f"{uid} {sdr} {sar} {sir}\n")

        if not sdr_scores:
            raise ValueError("No scores were computed. Please check the input data.")

        return {
            "SDR": float(np.mean(sdr_scores)),
            "SAR": float(np.mean(sar_scores)),
            "SIR": float(np.mean(sir_scores)),
        }
