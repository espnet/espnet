"""DNSMOS metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from tqdm import trange

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.utils.download_utils import download_url, setup_logger


class DNSMOS(BaseMetric):
    """Compute DNSMOS scores for hypothesis speech.

    This metric expects extracted speech as input,
    and produces non-intrusive scores (OVRL, SIG, BAK, P808_MOS) as output.

    Reference:
        [1] Chandan KA Reddy, Vishak Gopal, Ross Cutler.
            DNSMOS: A non-intrusive perceptual objective speech quality metric to
            evaluate noise suppressors.
            in Proc. IEEE ICASSP, 2021, pp. 6493-6497.
        [2] Chandan KA Reddy, Vishak Gopal, Ross Cutler.
            DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to
            evaluate noise suppressors.
            in Proc. IEEE ICASSP, 2022, pp. 886-890.
    """

    def __init__(
        self,
        ref_key: str = "ref",  # not used (for compatibility)
        hyp_key: str = "inf",
        batch_size: int = 1,
        ref_channel: int = 0,
        dnsmos_dir: str = "",
        use_gpu: bool = False,
        convert_to_torch: bool = False,
    ) -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        assert isinstance(batch_size, int) and batch_size > 0, batch_size
        self.batch_size = batch_size
        self.ref_channel = ref_channel
        self.device = "cuda" if use_gpu else "cpu"
        self.dnsmos_model = None
        self.dnsmos_dir = dnsmos_dir
        self.use_gpu = use_gpu
        self.convert_to_torch = convert_to_torch

    def _ensure_dnsmos(self):
        if self.dnsmos_model is None:
            from espnet2.enh.layers.dnsmos import DNSMOS_local

            logger = setup_logger(name="DNSMOS Metric")
            # Ensure that the DNSMOS models are downloaded and available
            url = (
                "https://github.com/microsoft/DNS-Challenge/"
                "raw/refs/heads/master/DNSMOS/DNSMOS/"
            )
            p835_model = f"{self.dnsmos_dir}/sig_bak_ovr.onnx"
            if not Path(p835_model).exists():
                logger.info(f"Downloading '{url}' to '{p835_model}'")
                download_url(
                    url=f"{url}/sig_bak_ovr.onnx",
                    dst_path=p835_model,
                    logger=logger,
                    step_percent=5,
                )
            p808_model = f"{self.dnsmos_dir}/model_v8.onnx"
            if not Path(p808_model).exists():
                logger.info(f"Downloading '{url}' to '{p808_model}'")
                download_url(
                    url=f"{url}/model_v8.onnx",
                    dst_path=p808_model,
                    logger=logger,
                    step_percent=5,
                )
            self.dnsmos_model = DNSMOS_local(
                primary_model_path=p835_model,
                p808_model_path=p808_model,
                use_gpu=self.use_gpu,
                convert_to_torch=self.convert_to_torch,
            )
        return self.dnsmos_model

    def load_audios(self, inf_batch: List[str]) -> List[Tuple[int, Tensor]]:
        inf_audios = []
        for inf_path in inf_batch:
            inf_audio, sr = sf.read(inf_path, dtype="float32")
            inf_audio = torch.as_tensor(inf_audio, device=self.device)
            inf_audios.append((sr, inf_audio))
        return inf_audios

    def compute_dnsmos(self, inf_audio: Tensor, sample_rate: int) -> Dict[str, float]:
        model = self._ensure_dnsmos()
        score = model(inf_audio, sample_rate)
        return {
            "OVRL": float(score["OVRL"]),
            "SIG": float(score["SIG"]),
            "BAK": float(score["BAK"]),
            "P808_MOS": float(score["P808_MOS"]),
        }

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for uid, row in self.iter_inputs(data, self.hyp_key):
            samples.append((uid, row[self.hyp_key]))

        scores = []
        num_lines = 0
        if (test_dir / "dnsmos").exists() and (test_dir / "dnsmos").is_file():
            with (test_dir / "dnsmos").open("rb") as f:
                num_lines = sum(1 for _ in f)
        if num_lines == len(samples):
            print(
                f"Found existing DNSMOS scores in '{test_dir / 'dnsmos'}', "
                "skipping computation and loading scores from file."
            )
            with (test_dir / "dnsmos").open("r", encoding="utf-8") as f:
                for line in f:
                    uid, ovr, sig, bak, p808_mos = line.strip().split()
                    scores.append(
                        {
                            "OVRL": float(ovr),
                            "SIG": float(sig),
                            "BAK": float(bak),
                            "P808_MOS": float(p808_mos),
                        }
                    )
        else:
            with (test_dir / "dnsmos").open("w", encoding="utf-8") as f:
                for i in trange(0, len(samples), self.batch_size):
                    batch = samples[i : i + self.batch_size]
                    batch_uids = [sample[0] for sample in batch]
                    inf_paths = [sample[1] for sample in batch]

                    inf_audios = self.load_audios(inf_paths)
                    for uid, inf_item in zip(batch_uids, inf_audios):
                        sample_rate = inf_item[0]
                        inf_audio = inf_item[1]
                        score = self.compute_dnsmos(inf_audio, sample_rate)
                        scores.append(score)
                        f.write(
                            f"{uid} {score['OVRL']} {score['SIG']} {score['BAK']} "
                            f"{score['P808_MOS']}\n"
                        )

        if not scores:
            raise ValueError("No scores were computed. Please check the input data.")

        return {
            "DNSMOS_OVRL": float(np.mean([s["OVRL"] for s in scores])),
            "DNSMOS_SIG": float(np.mean([s["SIG"] for s in scores])),
            "DNSMOS_BAK": float(np.mean([s["BAK"] for s in scores])),
            "DNSMOS_P808_MOS": float(np.mean([s["P808_MOS"] for s in scores])),
        }
