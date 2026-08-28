"""STOI and ESTOI metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from tqdm import trange

from espnet3.components.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
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
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "inf",
        batch_size: int = 1,
        ref_channel: int = 0,
    ) -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        assert isinstance(batch_size, int) and batch_size > 0, batch_size
        self.batch_size = batch_size
        self.ref_channel = ref_channel
        self.device = "cpu"
        self.name = "STOI"
        self.extended = False

    def _ensure_stoi(self):
        try:
            from pystoi import stoi as pystoi_stoi
        except ImportError as exc:
            raise RuntimeError("'pystoi' is required to compute STOI/ESTOI") from exc
        return pystoi_stoi

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

    def compute_stoi(
        self, ref_audio: Tensor, inf_audio: Tensor, sample_rate: int
    ) -> Dict[str, float]:
        stoi_fn = self._ensure_stoi()
        score = stoi_fn(
            ref_audio.cpu().numpy(),
            inf_audio.cpu().numpy(),
            fs_sig=sample_rate,
            extended=self.extended,
        )
        return float(score * 100.0)

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        pairs = []
        for uid, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            pairs.append((uid, row[self.ref_key], row[self.hyp_key]))

        scores = []
        num_lines = 0
        name = "estoi" if self.extended else "stoi"
        if (test_dir / name).exists() and (test_dir / name).is_file():
            with (test_dir / name).open("rb") as f:
                num_lines = sum(1 for _ in f)
        if num_lines == len(pairs):
            print(
                f"Found existing {name.upper()} scores in '{test_dir / name}', "
                "skipping computation and loading scores from file."
            )
            with (test_dir / name).open("r", encoding="utf-8") as f:
                for line in f:
                    uid, score = line.strip().split()
                    scores.append(float(score))
        else:
            with (test_dir / name).open("w", encoding="utf-8") as f:
                for i in trange(0, len(pairs), self.batch_size):
                    batch = pairs[i : i + self.batch_size]
                    batch_uids = [sample[0] for sample in batch]
                    ref_paths = [sample[1] for sample in batch]
                    inf_paths = [sample[2] for sample in batch]

                    ref_audios, inf_audios = self.load_audio_pairs(ref_paths, inf_paths)
                    for uid, ref_item, inf_item in zip(
                        batch_uids, ref_audios, inf_audios
                    ):
                        sample_rate = ref_item[0]
                        ref_audio = ref_item[1]
                        inf_audio = inf_item[1]
                        score = self.compute_stoi(ref_audio, inf_audio, sample_rate)
                        scores.append(score)
                        f.write(f"{uid} {score}\n")

        if not scores:
            raise ValueError("No scores were computed. Please check the input data.")

        return {self.name: float(np.mean(scores))}


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
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "inf",
        batch_size: int = 1,
        ref_channel: int = 0,
    ) -> None:
        super().__init__(
            ref_key=ref_key,
            hyp_key=hyp_key,
            batch_size=batch_size,
            ref_channel=ref_channel,
        )
        self.name = "ESTOI"
        self.extended = True
