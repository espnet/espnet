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

    score_config can be either:
      - a string/path to an existing YAML file consumed directly by VERSA, or
      - a list of metric dicts (inline) — materialized inside the per-test-set
        scoring directory next to result.json / avg_result.json.
    """

    def __init__(
        self,
        score_config: Union[str, Path, List[Dict[str, Any]]],
        hyp_key: str = "hyp",
        ref_key: str = "ref",
        text_key: str = "text",
        use_gpu: bool = True,
        io: str = "soundfile",
    ) -> None:
        # OmegaConf ListConfig → plain Python list (json/yaml safe)
        if OmegaConf.is_list(score_config):
            score_config = OmegaConf.to_container(score_config, resolve=True)
        self._score_config_input = score_config   # path string or list; materialized in __call__

        self.hyp_key = hyp_key
        self.ref_key = ref_key
        self.text_key = text_key
        self.use_gpu = use_gpu
        self.io = io

    def _resolve_score_config_path(self, eval_dir: Path) -> Path:
        """Return a YAML file path VERSA can `open()`.

        - If the input is a path-like that exists on disk, use it verbatim.
        - If it's an inline list, dump to ``eval_dir/score_config.yaml``.
        """
        sc = self._score_config_input

        if isinstance(sc, (str, Path)):
            p = Path(sc)
            if not p.is_file():
                raise FileNotFoundError(f"VERSA score_config path not found: {p}")
            return p

        if isinstance(sc, list):
            out = eval_dir / "score_config.yaml"
            with out.open("w", encoding="utf-8") as f:
                yaml.safe_dump(sc, f, sort_keys=False)
            logger.info("Wrote inline VERSA metric list to %s", out)
            return out

        raise TypeError(
            f"score_config must be a path or a list, got {type(sc)}"
        )

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        if self.hyp_key not in data:
            raise KeyError(
                f"VersaMetric requires '{self.hyp_key}' input. "
                f"Got: {list(data.keys())}"
            )
        if self.ref_key not in data:
            raise KeyError(
                f"VersaMetric requires '{self.ref_key}' input. "
                f"Got: {list(data.keys())}"
            )

        eval_dir = Path(inference_dir) / test_name / "scoring" / "versa_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        score_config_path = self._resolve_score_config_path(eval_dir)
        result_file = eval_dir / "result.json"

        cmd = [
            "python", "-m", "versa.bin.scorer",
            "--pred", str(data[self.hyp_key]),
            "--gt", str(data[self.ref_key]),
            "--score_config", str(score_config_path),
            "--cache_folder", str(eval_dir / "cache"),
            "--output_file", str(result_file),
            "--io", self.io,
        ]
        if self.text_key in data:
            cmd += ["--text", str(data[self.text_key])]
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
