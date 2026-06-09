"""VERSA-based TTS metric wrapper for the measure stage."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from omegaconf import OmegaConf

from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


class VersaMetric(BaseMetric):
    """Wrap versa.bin.scorer.

    The config of the versa metrics must be provided in YAML files.
    This will dump the versa config to the experiment directory.
    """

    def __init__(
        self,
        score_config: Union[str, Path, List[Dict[str, Any]]],
        wav_key: str = "wav",
        ref_key: str = "ref",
        text_key: str = "text",
        use_gpu: bool = True,
        io: str = "soundfile",
    ) -> None:
        self.score_config = OmegaConf.to_container(score_config, resolve=True)
        self.wav_key = wav_key
        self.ref_key = ref_key
        self.text_key = text_key
        self.use_gpu = use_gpu
        self.io = io

    def _resolve_score_config_path(self, eval_dir: Path) -> Path:
        """Return a YAML file path VERSA can `open()`.

        Read the score_config and dump it to file inside ``eval_dir/versa_config.yaml``
        """
        out = eval_dir / "versa_config.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.score_config, f, sort_keys=False)
        logger.info("Wrote inline VERSA metric list to %s", out)
        return out


    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        if self.wav_key not in data:
            raise KeyError(
                f"VersaMetric requires '{self.wav_key}' input. "
                f"Got: {list(data.keys())}"
            )
        if self.ref_key not in data:
            raise KeyError(
                f"VersaMetric requires '{self.ref_key}' input. "
                f"Got: {list(data.keys())}"
            )
        if self.text_key in data:
            raise KeyError(
                f"VersaMetric requires '{self.text_key}' input. "
                f"Got: {list(data.keys())}"
            )

        eval_dir = Path(inference_dir) / test_name / "scoring" / "versa_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        score_config_path = self._resolve_score_config_path(eval_dir)
        result_file = eval_dir / "result.json"

        cmd = [
            "python",
            "-m",
            "versa.bin.scorer",
            "--pred",
            str(data[self.wav_key]),
            "--gt",
            str(data[self.ref_key]),
            "--text",
            str(data[self.text_key]),
            "--score_config",
            str(score_config_path),
            "--cache_folder",
            str(eval_dir / "cache"),
            "--output_file",
            str(result_file),
            "--io",
            self.io,
        ]
        if self.use_gpu:
            cmd.append("--use_gpu")

        logger.info("Running VERSA: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

        averages = self._aggregate(result_file)
        avg_path = eval_dir / "avg_result.json"
        with avg_path.open("w") as f:
            json.dump(averages, f, indent=2)
        logger.info(
            "Wrote VERSA averages for '%s' to %s (%d metrics)",
            test_name,
            avg_path,
            len(averages),
        )
        return averages

    @staticmethod
    def _aggregate(result_file: Path) -> Dict[str, float]:
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        with result_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for key, value in record.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        sums[key] = sums.get(key, 0.0) + float(value)
                        counts[key] = counts.get(key, 0) + 1
        return {key: round(sums[key] / counts[key], 4) for key in sums}
