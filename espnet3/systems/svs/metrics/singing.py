"""Objective singing voice synthesis metrics."""

from __future__ import annotations

import logging
from math import log2
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
from scipy import spatial

from espnet3.components.metrics.base_metric import BaseMetric

try:
    import pysptk
    import pyworld as pw
    from fastdtw import fastdtw
except ImportError:  # pragma: no cover - reported at call time
    pysptk = pw = fastdtw = None

logger = logging.getLogger(__name__)

METRIC_KEYS = ("mcd", "log_f0_rmse", "semitone_acc", "vuv_err")

# The 12 pitch-class names used to bin F0 into semitones (C0 = A4 * 2^-4.75).
_C0 = 440.0 * 2 ** (-4.75)


def _best_mcep_params(fs: int) -> Tuple[int, float]:
    """Return the (order, alpha) mel-cepstrum setting for a sampling rate."""
    params = {
        16000: (23, 0.42),
        22050: (34, 0.45),
        24000: (34, 0.46),
        44100: (39, 0.53),
        48000: (39, 0.55),
    }
    if fs not in params:
        raise ValueError(f"No mel-cepstrum setting for fs={fs}.")
    return params[fs]


def _hz_to_semitone(f0: float) -> int:
    """Bin one F0 value to a semitone index; 0 Hz (unvoiced) maps to -1."""
    if f0 == 0:
        return -1
    return int(round(12 * log2(f0 / _C0)))


def _sptk_mcep(
    x: np.ndarray, n_fft: int, n_shift: int, mcep_dim: int, mcep_alpha: float
) -> np.ndarray:
    """Extract frame-wise SPTK mel-cepstrum from an int16 waveform."""
    n_frame = (len(x) - n_fft) // n_shift + 1
    win = pysptk.sptk.hamming(n_fft)
    mcep = [
        pysptk.mcep(
            x[n_shift * i : n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]
    return np.stack(mcep)


def _world_mcep_f0(
    x: np.ndarray,
    fs: int,
    f0min: int,
    f0max: int,
    n_fft: int,
    n_shift: int,
    mcep_dim: int,
    mcep_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract WORLD F0 and the mel-cepstrum of its spectral envelope."""
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x, fs, f0_floor=f0min, f0_ceil=f0max, frame_period=n_shift / fs * 1000
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    return pysptk.sp2mc(sp, mcep_dim, mcep_alpha), f0


def _dtw_paths(gen: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align two feature sequences with DTW and return the index paths."""
    _, path = fastdtw(gen, ref, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    return twf[0], twf[1]


def _score_pair(args) -> Tuple[str, Dict[str, float]]:
    """Compute the four metrics for one (generated, reference) wav pair."""
    utt_id, gen_path, ref_path, f0min, f0max, n_fft, n_shift, mcep_dim, alpha = args
    gen_x, fs = sf.read(gen_path, dtype="int16")
    ref_x, ref_fs = sf.read(ref_path, dtype="int16")
    if fs != ref_fs:
        raise ValueError(f"{utt_id}: sampling rate mismatch ({fs} vs {ref_fs})")
    if mcep_dim is None or alpha is None:
        mcep_dim, alpha = _best_mcep_params(fs)

    # MCD on SPTK mel-cepstrum, as in evaluate_mcd.py.
    gen_mcep = _sptk_mcep(gen_x, n_fft, n_shift, mcep_dim, alpha)
    ref_mcep = _sptk_mcep(ref_x, n_fft, n_shift, mcep_dim, alpha)
    gen_idx, ref_idx = _dtw_paths(gen_mcep, ref_mcep)
    diff2sum = np.sum((gen_mcep[gen_idx] - ref_mcep[ref_idx]) ** 2, 1)
    mcd = float(np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum)))

    # F0-based metrics on WORLD features, as in evaluate_{f0,semitone,vuv}.py.
    world_args = (f0min, f0max, n_fft, n_shift, mcep_dim, alpha)
    gen_mcep, gen_f0 = _world_mcep_f0(gen_x, fs, *world_args)
    ref_mcep, ref_f0 = _world_mcep_f0(ref_x, fs, *world_args)
    gen_idx, ref_idx = _dtw_paths(gen_mcep, ref_mcep)
    gen_f0, ref_f0 = gen_f0[gen_idx], ref_f0[ref_idx]

    # log-F0 RMSE is defined over frames voiced in both; NaN when there are none.
    voiced = (gen_f0 != 0) & (ref_f0 != 0)
    if voiced.any():
        log_f0_rmse = float(
            np.sqrt(np.mean((np.log(gen_f0[voiced]) - np.log(ref_f0[voiced])) ** 2))
        )
    else:
        log_f0_rmse = float("nan")
    gen_semitone = np.array([_hz_to_semitone(f) for f in gen_f0])
    ref_semitone = np.array([_hz_to_semitone(f) for f in ref_f0])
    semitone_acc = float(np.mean(gen_semitone == ref_semitone))
    vuv_err = float(np.mean((gen_f0 != 0) != (ref_f0 != 0)))

    return utt_id, dict(
        mcd=mcd, log_f0_rmse=log_f0_rmse, semitone_acc=semitone_acc, vuv_err=vuv_err
    )


class SingingMetrics(BaseMetric):
    """MCD, log-F0 RMSE, semitone accuracy and V/UV error of synthesized singing.

    A port of ``egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_{mcd,f0,semitone,
    vuv}.py``: generated and reference waveforms are aligned with DTW on their
    mel-cepstra, MCD is computed on SPTK mel-cepstrum, and the F0 metrics on
    WORLD (harvest) F0 over the aligned frames. Semitone accuracy and V/UV
    error are the standard SVS numbers in ``egs2/TEMPLATE/svs1``.

    Example:
        ```yaml
        metrics:
          - metric:
              _target_: espnet3.systems.svs.metrics.singing.SingingMetrics
              num_workers: 16
            inputs:
              wav: wav
              ref: ref
        ```
    """

    def __init__(
        self,
        wav_key: str = "wav",
        ref_key: str = "ref",
        f0min: int = 40,
        f0max: int = 800,
        n_fft: int = 1024,
        n_shift: int = 256,
        mcep_dim: int | None = None,
        mcep_alpha: float | None = None,
        num_workers: int = 1,
    ) -> None:
        """Initialize the metric.

        Args:
            wav_key: Input alias of the generated wav SCP.
            ref_key: Input alias of the reference wav SCP.
            f0min: Minimum F0 for WORLD harvest.
            f0max: Maximum F0 for WORLD harvest.
            n_fft: FFT size for the mel-cepstrum analysis.
            n_shift: Frame shift in samples.
            mcep_dim: Mel-cepstrum order. ``None`` picks it from the sampling
                rate, as the egs2 scripts do.
            mcep_alpha: All-pass constant. ``None`` picks it from the sampling
                rate.
            num_workers: Processes used to score utterances in parallel.
        """
        self.wav_key = wav_key
        self.ref_key = ref_key
        self.f0min = f0min
        self.f0max = f0max
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.mcep_dim = mcep_dim
        self.mcep_alpha = mcep_alpha
        self.num_workers = num_workers

    def _ensure_deps(self) -> None:
        if pysptk is None or pw is None or fastdtw is None:
            raise RuntimeError(
                "pyworld, pysptk and fastdtw are required for SingingMetrics. "
                "Please install them with `pip install espnet[tts] pysptk fastdtw`."
            )

    def _jobs(self, data: Dict[str, Path]) -> List[tuple]:
        jobs = []
        for utt_id, row in self.iter_inputs(data, self.wav_key, self.ref_key):
            jobs.append(
                (
                    utt_id,
                    row[self.wav_key],
                    row[self.ref_key],
                    self.f0min,
                    self.f0max,
                    self.n_fft,
                    self.n_shift,
                    self.mcep_dim,
                    self.mcep_alpha,
                )
            )
        return jobs

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Score every utterance and return the corpus-level averages.

        Args:
            data: Mapping of input aliases to SCP files. ``data[wav_key]`` and
                ``data[ref_key]`` list generated and reference wav paths with
                aligned utterance ids.
            test_name: Test set name, used for the per-utterance output file.
            inference_dir: Directory holding the inference outputs; the
                per-utterance scores are written to
                ``<inference_dir>/<test_name>/singing_metrics.txt``.

        Returns:
            ``mcd``, ``log_f0_rmse``, ``semitone_acc`` and ``vuv_err`` averaged
            over the test set. Utterances without a frame voiced in both wavs
            have no log-F0 RMSE and are left out of that average.
        """
        self._ensure_deps()
        jobs = self._jobs(data)
        if self.num_workers > 1:
            with Pool(self.num_workers) as pool:
                results: Iterable = pool.map(_score_pair, jobs)
        else:
            results = map(_score_pair, jobs)
        results = list(results)

        out_path = Path(inference_dir) / test_name / "singing_metrics.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("utt_id\t" + "\t".join(METRIC_KEYS) + "\n")
            for utt_id, scores in results:
                values = "\t".join(f"{scores[k]:.4f}" for k in METRIC_KEYS)
                fh.write(f"{utt_id}\t{values}\n")

        summary = {
            key: float(np.nanmean([scores[key] for _, scores in results]))
            for key in METRIC_KEYS
        }
        logger.info("%s: %s", test_name, summary)
        return summary
