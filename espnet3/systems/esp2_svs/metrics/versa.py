"""VERSA metrics for synthesized singing."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from omegaconf import OmegaConf

try:
    import versa
except ImportError:
    versa = None

from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


class VersaMetric(BaseMetric):
    """Score ``infer`` outputs with VERSA.

    Runs ``versa.bin.scorer`` (https://github.com/wavlab-speech/versa) on the
    synthesized waveforms and returns the average of every numeric score it
    reports. As in ``egs2/TEMPLATE/svs1/svs.sh``, the wav SCP is split into
    ``nj`` shards scored by separate jobs; with ``use_gpu``, job ``i`` runs on
    GPU ``i % num_gpus``. Per-utterance scores and job logs are written to
    ``<output_dir>/<test_name>/versa/``.

    Example:
        The default of ``egs3/TEMPLATE/esp2_svs/conf/metrics.yaml``:

        .. code-block:: yaml

            metrics:
              - metric:
                  _target_: espnet3.systems.esp2_svs.metrics.versa.VersaMetric
                  score_config:
                    - name: mcd_f0
                      f0min: 40
                      f0max: 800
                      dtw: true
                    - name: pseudo_mos
                      predictor_types: [singmos]
                inputs:
                  wav: wav
                  ref: ref
    """

    def __init__(
        self,
        score_config: List[dict],
        wav_key: str = "wav",
        ref_key: Optional[str] = "ref",
        text_key: Optional[str] = None,
        nj: int = 1,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the VERSA metric.

        Args:
            score_config: VERSA metric definitions, as in a VERSA config file.
            wav_key: Input alias of the synthesized wav SCP.
            ref_key: Input alias of the reference wav SCP, or ``None`` when no
                metric needs a reference.
            text_key: Input alias of a transcript SCP, or ``None`` when no
                metric needs text.
            nj: Number of scoring jobs.
            use_gpu: Run the jobs on GPU.
        """
        if OmegaConf.is_config(score_config):
            score_config = OmegaConf.to_container(score_config, resolve=True)
        self.score_config = score_config
        self.wav_key = wav_key
        self.ref_key = ref_key
        self.text_key = text_key
        self.nj = nj
        self.use_gpu = use_gpu

    def _ensure_versa(self) -> None:
        """Raise an error if VERSA is not installed.

        Raises:
            RuntimeError: If ``versa`` is not installed.
        """
        if versa is None:
            raise RuntimeError(
                "VERSA is required to compute VersaMetric. "
                "Please install it with `cd tools && make versa.done`."
            )

    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, float]:
        """Score one test set and return the average of each VERSA score.

        Args:
            data: Mapping of input aliases to the SCP files written by
                ``infer``.
            test_name: Test set name.
            output_dir: Inference directory. Results are written to
                ``<output_dir>/<test_name>/versa/``.

        Returns:
            The average of every numeric field VERSA reports, e.g. ``mcd``,
            ``f0rmse``, ``f0corr`` and ``singmos``.

        Raises:
            RuntimeError: If VERSA is not installed, a job fails or no
                utterance is scored.
        """
        self._ensure_versa()
        # VERSA saves the models it downloads in its working directory, so the
        # jobs run in eval_dir and every path they get is absolute.
        eval_dir = (Path(output_dir) / test_name / "versa").resolve()
        eval_dir.mkdir(parents=True, exist_ok=True)
        options = self._write_shared_inputs(data, eval_dir)

        utterances = self._read_wav_scp(data, self.wav_key)
        nj = min(self.nj, len(utterances))
        if nj > 1:
            # Score one utterance first, so that the models are downloaded
            # once instead of by every job at the same time.
            self._run_jobs(eval_dir, {"warmup": utterances[:1]}, options)

        shards = {}
        for rank in range(nj):
            shards[str(rank)] = utterances[rank::nj]
        logger.info("VERSA: %d utterances, %d job(s)", len(utterances), nj)
        self._run_jobs(eval_dir, shards, options)

        scores = []
        for name in shards:
            with open(eval_dir / f"result.{name}.jsonl", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        scores.append(json.loads(line))
        if len(scores) == 0:
            raise RuntimeError(f"VERSA scored no utterance for {test_name}")

        averages = {}
        for key, value in scores[0].items():
            if isinstance(value, (int, float)):
                values = [score[key] for score in scores]
                averages[key] = float(np.nanmean(values))
        return averages

    def _read_wav_scp(self, data: Dict[str, Path], key: str) -> List[str]:
        """Read the wav SCP ``data[key]`` as lines with absolute wav paths."""
        lines = []
        for utt_id, row in self.iter_inputs(data, key):
            lines.append(f"{utt_id} {Path(row[key]).resolve()}\n")
        return lines

    def _write_shared_inputs(self, data: Dict[str, Path], eval_dir: Path) -> List[str]:
        """Write the inputs every job reads and return their scorer options."""
        config_path = eval_dir / "score_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.score_config, f)
        options = ["--score_config", str(config_path), "--io", "soundfile"]

        if self.ref_key is not None:
            gt_path = eval_dir / "gt.scp"
            with open(gt_path, "w", encoding="utf-8") as f:
                f.writelines(self._read_wav_scp(data, self.ref_key))
            options += ["--gt", str(gt_path)]
        if self.text_key is not None:
            options += ["--text", str(Path(data[self.text_key]).resolve())]
        if self.use_gpu:
            options.append("--use_gpu")
        return options

    @staticmethod
    def _run_jobs(
        eval_dir: Path, shards: Dict[str, List[str]], options: List[str]
    ) -> None:
        """Score each shard with its own ``versa.bin.scorer`` process."""
        processes = {}
        for rank, (name, lines) in enumerate(shards.items()):
            pred_path = eval_dir / f"pred.{name}.scp"
            with open(pred_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            command = [sys.executable, "-m", "versa.bin.scorer"]
            command += ["--pred", str(pred_path), "--rank", str(rank)]
            command += ["--output_file", f"result.{name}.jsonl"] + options
            with open(eval_dir / f"versa.{name}.log", "w") as log:
                processes[name] = subprocess.Popen(
                    command,
                    cwd=eval_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    # One thread per job: the jobs share the CPUs.
                    env={**os.environ, "OMP_NUM_THREADS": "1"},
                )

        failed = []
        for name, process in processes.items():
            if process.wait() != 0:
                failed.append(name)
        if failed:
            raise RuntimeError(
                f"VERSA failed on shard(s) {', '.join(failed)}; "
                f"see {eval_dir}/versa.*.log"
            )
