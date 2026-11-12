"""DNSMOS metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from espnet3.contract import Field
from espnet3.systems.esp2_tse.src.audio_metrics import AudioMetric, read_audio
from espnet3.utils.download_utils import download_url, setup_logger


class DNSMOS(AudioMetric):
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

    inputs = (Field("hyp", "audio"),)
    outputs = (
        Field("DNSMOS_OVRL", "number"),
        Field("DNSMOS_SIG", "number"),
        Field("DNSMOS_BAK", "number"),
        Field("DNSMOS_P808_MOS", "number"),
    )

    def __init__(
        self,
        batch_size: int = 1,
        ref_channel: int = 0,
        dnsmos_dir: str = ".",
        use_gpu: bool = False,
        convert_to_torch: bool = False,
    ) -> None:
        """Initialize the DNSMOS measure.

        Args:
            batch_size: batch size for batched inference.
            ref_channel: Reference channel index for aligning multi-channel signals.
            dnsmos_dir: Directory to store the DNSMOS model.
            use_gpu: Whether to use GPU for inference.
            convert_to_torch: Whether to convert the model to TorchScript for faster
                              inference.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.ref_channel = ref_channel
        self.device = "cuda" if use_gpu else "cpu"
        self.dnsmos_model = None
        self.dnsmos_dir = dnsmos_dir
        self.use_gpu = use_gpu
        self.convert_to_torch = convert_to_torch

        self._init_scores()

    def _ensure_dnsmos(self):
        """Load the DNSMOS model if it is not already loaded."""
        if self.dnsmos_model is None:
            from espnet2.enh.layers.dnsmos import DNSMOS_local

            logger = setup_logger(name="DNSMOS Metric")
            # Ensure that the DNSMOS models are downloaded and available
            url = (
                "https://github.com/microsoft/DNS-Challenge/"
                "raw/refs/heads/master/DNSMOS/DNSMOS/"
            )
            p835_model = Path(self.dnsmos_dir) / "sig_bak_ovr.onnx"
            if not Path(p835_model).exists():
                logger.info(f"Downloading '{url}' to '{p835_model}'")
                download_url(
                    url=f"{url}/sig_bak_ovr.onnx",
                    dst_path=p835_model,
                    logger=logger,
                    step_percent=5,
                )
            p808_model = Path(self.dnsmos_dir) / "model_v8.onnx"
            if not Path(p808_model).exists():
                logger.info(f"Downloading '{url}' to '{p808_model}'")
                download_url(
                    url=f"{url}/model_v8.onnx",
                    dst_path=p808_model,
                    logger=logger,
                    step_percent=5,
                )
            self.dnsmos_model = DNSMOS_local(
                primary_model_path=str(p835_model),
                p808_model_path=str(p808_model),
                use_gpu=self.use_gpu,
                convert_to_torch=self.convert_to_torch,
            )
        return self.dnsmos_model

    def _load_audios(self, inf_batch: List[str]) -> List[Tuple[int, Tensor]]:
        """Load the input audio for inference."""
        inf_audios = []
        for inf_path in inf_batch:
            inf_audio, sr = read_audio(inf_path)
            inf_audio = torch.as_tensor(inf_audio, device=self.device)
            inf_audios.append((sr, inf_audio))
        return inf_audios

    def _compute_dnsmos(self, inf_audio: Tensor, sample_rate: int) -> Dict[str, float]:
        """Compute DNSMOS scores for a single inference audio."""
        model = self._ensure_dnsmos()
        score = model(inf_audio, sample_rate)
        return {
            "OVRL": float(score["OVRL"]),
            "SIG": float(score["SIG"]),
            "BAK": float(score["BAK"]),
            "P808_MOS": float(score["P808_MOS"]),
        }

    def update(self, *, hyp):
        """Accumulate DNSMOS scores for a batch of hypothesis audio."""
        for rate, audio in self._load_audios(hyp):
            score = self._compute_dnsmos(audio, rate)
            self._add_scores(
                {
                    f"DNSMOS_{key}": score[key]
                    for key in ("OVRL", "SIG", "BAK", "P808_MOS")
                }
            )
