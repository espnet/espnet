"""VERSA-based TTS metric wrapper for the measure stage."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


class VersaMetric(BaseMetric):
    """Run versa.bin.scorer on hypothesis / reference / text SCP files.

    The infer stage produces:
      - inference_dir/<test_name>/wav.scp   (synthesized audio)
      - inference_dir/<test_name>/ref.scp   (ground-truth audio paths)
      - inference_dir/<test_name>/text.scp  (reference transcripts)

    The metrics.yaml `inputs:` mapping translates those filenames into the
    aliases this class consumes ("hyp", "ref", "text"). Outputs land at:
      - <test_dir>/scoring/versa_eval/result.json     (per-utterance scores)
      - <test_dir>/scoring/versa_eval/avg_result.json (corpus averages)

    Works with any VERSA score_config — this wrapper forwards whatever
    inputs are present and VERSA dispatches internally.
    """

    def __init__(
        self,
        score_config: dict,
        hyp_key: str = "wav",
        ref_key: str = "ref",
        text_key: str = "text",
        use_gpu: bool = True,
    ) -> None:
        self.score_config = score_config
        self.hyp_key = hyp_key
        self.ref_key = ref_key
        self.text_key = text_key
        self.use_gpu = use_gpu

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
        if self.text_key not in data:
            raise KeyError(
                f"VersaMetric requires '{self.text_key}' input. "
                f"Got: {list(data.keys())}"
            )

        eval_dir = Path(inference_dir) / test_name / "scoring" / "versa_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        result_file = eval_dir / "result.json"
        cmd = [
            "python", "-m", "versa.bin.scorer",
            "--pred", str(data[self.hyp_key]),
            "--gt", str(data[self.ref_key]),
            "--text", str(data[self.text_key]),
            "--score_config", json.dumps(self.score_config),
            "--cache_folder", str(eval_dir / "cache"),
            "--output_file", str(result_file),
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
        """Average per-utterance numeric fields into corpus-level scores."""
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
