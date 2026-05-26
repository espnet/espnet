# egs3/vctk_noisy/enh/src/metrics.py
"""Speech enhancement metrics for VCTK-Noisy recipes.

Metrics:
  - SI-SNR  (scale-invariant signal-to-noise ratio)
  - PESQ    (requires pesq package: pip install pesq)
  - STOI    (requires pystoi package: pip install pystoi)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import resampy

from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

try:
    from pesq import pesq as pesq_fn
    _PESQ_AVAILABLE = True
except ImportError:
    _PESQ_AVAILABLE = False

try:
    from pystoi import stoi as stoi_fn
    _STOI_AVAILABLE = True
except ImportError:
    _STOI_AVAILABLE = False


def _si_snr(ref: np.ndarray, est: np.ndarray) -> float:
    """Compute SI-SNR between reference and estimate signals."""
    min_len = min(len(ref), len(est))
    ref = ref[:min_len] - ref[:min_len].mean()
    est = est[:min_len] - est[:min_len].mean()
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8) * ref
    e_noise = est - s_target
    return float(
        10.0 * np.log10((np.dot(s_target, s_target) + 1e-8) / (np.dot(e_noise, e_noise) + 1e-8))
    )

def _load_clean_map(manifest_dir: Path, test_name: str) -> dict[str, Path]:
    """Load {utt_id: clean_wav_path} from the TSV manifest for test_name."""
    # test_name is one of: train, valid, test
    # For measure stage, only valid and test are evaluated.
    manifest_path = manifest_dir / f"{test_name}.tsv"
    assert manifest_path.is_file(), f"Manifest not found: {manifest_path}"
    clean_map: dict[str, Path] = {}
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, _noisy, clean = line.split("\t", 2)
            clean_map[utt_id] = Path(clean)
    return clean_map



class SISNRMetric(BaseMetric):
    """SI-SNR metric over enhanced wav artifacts.

    Reads enhanced.scp (wav paths from inference) and matches clean
    references from the TSV manifest.

    Args:
        hyp_key: SCP key for enhanced wav paths. Defaults to ``"enhanced"``.
        manifest_dir: Directory containing train/valid/test.tsv manifests.
    """

    def __init__(self, hyp_key: str = "enhanced", manifest_dir: str | None = None):
        self.hyp_key = hyp_key
        self.manifest_dir = Path(manifest_dir) if manifest_dir else None

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        assert self.manifest_dir is not None, "manifest_dir must be set"
        clean_map = _load_clean_map(self.manifest_dir, test_name)

        scores: list[float] = []
        for utt_id, row in self.iter_inputs(data, self.hyp_key):
            # row[self.hyp_key] is the wav file path (value column in enhanced.scp)
            enh_wav, enh_sr = sf.read(row[self.hyp_key])
            if utt_id not in clean_map:
                logger.warning("utt_id %s not in manifest, skipping.", utt_id)
                continue
            ref_wav, ref_sr = sf.read(str(clean_map[utt_id]))
            if ref_sr != enh_sr:
                ref_wav = resampy.resample(ref_wav, ref_sr, enh_sr)    
            scores.append(_si_snr(ref_wav.astype(np.float32), enh_wav.astype(np.float32)))

        mean = float(np.mean(scores)) if scores else float("nan")
        logger.info("[%s] SI-SNR: %.2f dB (%d utts)", test_name, mean, len(scores))
        return {"SI-SNR": round(mean, 4)}


class PESQMetric(BaseMetric):
    """PESQ (wide-band, 16 kHz) metric. Requires: pip install pesq.

    Args:
        hyp_key: SCP key for enhanced wav paths.
        manifest_dir: Directory containing train/valid/test.tsv manifests.
        fs: Sampling rate. Must be 8000 or 16000.
    """


    def __init__(
        self,
        hyp_key: str = "enhanced",
        manifest_dir: str | None = None,
        fs: int = 16000,
    ):
        self.hyp_key = hyp_key
        self.manifest_dir = Path(manifest_dir) if manifest_dir else None
        self.fs = fs

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        if not _PESQ_AVAILABLE:
            raise RuntimeError("pesq is required: pip install pesq")
        assert self.manifest_dir is not None
        clean_map = _load_clean_map(self.manifest_dir, test_name)
        mode = "wb" if self.fs == 16000 else "nb"
        scores: list[float] = []

        for utt_id, row in self.iter_inputs(data, self.hyp_key):
            enh_wav, enh_sr = sf.read(row[self.hyp_key])
            if enh_sr != self.fs:
                enh_wav = resampy.resample(enh_wav, enh_sr, self.fs)
            if utt_id not in clean_map:
                continue
            ref_wav, ref_sr = sf.read(str(clean_map[utt_id]))
            if ref_sr != self.fs:
                ref_wav = resampy.resample(ref_wav, ref_sr, self.fs)
            min_len = min(len(ref_wav), len(enh_wav))
            scores.append(pesq_fn(self.fs, ref_wav[:min_len], enh_wav[:min_len], mode))

        mean = float(np.mean(scores)) if scores else float("nan")
        logger.info("[%s] PESQ: %.4f (%d utts)", test_name, mean, len(scores))
        return {"PESQ": round(mean, 4)}

class STOIMetric(BaseMetric):
    """STOI metric. Requires: pip install pystoi.

    Args:
        hyp_key: SCP key for enhanced wav paths.
        manifest_dir: Directory containing train/valid/test.tsv manifests.
        fs: Sampling rate in Hz.
        extended: If True, compute extended STOI (ESTOI).
    """

    def __init__(
        self,
        hyp_key: str = "enhanced",
        manifest_dir: str | None = None,
        fs: int = 16000,
        extended: bool = False,
    ):
        self.hyp_key = hyp_key
        self.manifest_dir = Path(manifest_dir) if manifest_dir else None
        self.fs = fs
        self.extended = extended

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        if not _STOI_AVAILABLE:
            raise RuntimeError("pystoi is required: pip install pystoi")
        assert self.manifest_dir is not None
        clean_map = _load_clean_map(self.manifest_dir, test_name)
        scores: list[float] = []

        for utt_id, row in self.iter_inputs(data, self.hyp_key):
            enh_wav, enh_sr = sf.read(row[self.hyp_key])
            if enh_sr != self.fs:
                enh_wav = resampy.resample(enh_wav, enh_sr, self.fs)
            if utt_id not in clean_map:
                continue
            ref_wav, ref_sr = sf.read(str(clean_map[utt_id]))
            if ref_sr != self.fs:
                ref_wav = resampy.resample(ref_wav, ref_sr, self.fs)
            min_len = min(len(ref_wav), len(enh_wav))
            scores.append(stoi_fn(ref_wav[:min_len], enh_wav[:min_len], self.fs, extended=self.extended))

        mean = float(np.mean(scores)) if scores else float("nan")
        label = "ESTOI" if self.extended else "STOI"
        logger.info("[%s] %s: %.4f (%d utts)", test_name, label, mean, len(scores))
        return {label: round(mean, 4)}