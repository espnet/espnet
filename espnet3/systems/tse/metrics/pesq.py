"""PESQ metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from tqdm import trange

from espnet3.components.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    """Compute PESQ scores for reference/hypothesis speech pairs.

    This metric expects extracted speech and reference speech as input,
    and produces an objective score as output.

    Note: Please make sure that you have the proper license to use PESQ.

    Reference:
        Antony W Rix, John G Beerends, Michael P Hollier, Andries P Hekstra.
        Perceptual evaluation of speech quality (PESQ)---a new method for
        speech quality assessment of telephone networks and codecs.
        in Proc. IEEE ICASSP, 2001, pp. 749-752.
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

    def _ensure_pesq(self):
        try:
            from pesq import PesqError, pesq
        except ImportError as exc:
            raise RuntimeError("'pesq' is required to compute PESQ") from exc
        return PesqError, pesq

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

    def compute_pesq(self, ref_audio: Tensor, inf_audio: Tensor, sample_rate: int):
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
        return float(score)

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        pairs = []
        for uid, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            pairs.append((uid, row[self.ref_key], row[self.hyp_key]))

        scores = []
        with (test_dir / "pesq").open("w", encoding="utf-8") as f:
            for i in trange(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                batch_uids = [sample[0] for sample in batch]
                ref_paths = [sample[1] for sample in batch]
                inf_paths = [sample[2] for sample in batch]

                ref_audios, inf_audios = self.load_audio_pairs(ref_paths, inf_paths)
                for uid, ref_item, inf_item in zip(batch_uids, ref_audios, inf_audios):
                    sample_rate = ref_item[0]
                    ref_audio = ref_item[1]
                    inf_audio = inf_item[1]
                    score = self.compute_pesq(ref_audio, inf_audio, sample_rate)
                    scores.append(score)
                    f.write(f"{uid} {score}\n")

        if not scores:
            raise ValueError("No scores were computed. Please check the input data.")
        return {"PESQ": float(np.mean(scores))}
