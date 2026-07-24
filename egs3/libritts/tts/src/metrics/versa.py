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
        score_config,
        wav_key: str = "wav",
        ref_key: str = "ref",
        text_key: str = "text",
        use_gpu: bool = True,
        io: str = "soundfile",
    ) -> None:
        self.score_config = score_config
        self.wav_key = wav_key
        self.ref_key = ref_key
        self.text_key = text_key
        self.use_gpu = use_gpu
        self.io = io

    def _resolve_score_config_path(self, eval_dir: Path) -> Path:
        """Return a YAML file path VERSA can `open()`.

        If ``score_config`` is a path to an existing file, return it directly.
        Otherwise treat it as an inline config object and dump it to
        ``eval_dir/versa_config.yaml``.
        """
        if isinstance(self.score_config, (str, Path)):
            p = Path(self.score_config)
            if not p.is_file():
                raise FileNotFoundError(f"VERSA score_config path does not exist: {p}")
            logger.info("Using VERSA config file %s", p)
            return p

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
        if self.text_key not in data:
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
        self.summarize(averages, test_name)
        return averages

    @staticmethod
    def summarize(scores: Dict[str, float], test_name: str = "") -> None:
        """Log a formatted summary of VERSA scores."""
        header = f"VERSA scores — {test_name}" if test_name else "VERSA scores"

        # Detect prefixes for WER/CER component groups (e.g. espnet_wer_, whisper_wer_)
        def _find_prefix(metric: str) -> str | None:
            """Return '<prefix>_<metric>_' if all four ops are present, else None."""
            for k in scores:
                if k.endswith(f"_{metric}_delete"):
                    prefix = k[: -(len(metric) + 8)]  # strip '_<metric>_delete'
                    ops = [
                        f"{prefix}_{metric}_{op}"
                        for op in ("delete", "insert", "replace", "equal")
                    ]
                    if all(op in scores for op in ops):
                        return f"{prefix}_{metric}_"
            return None

        wer_prefix = _find_prefix("wer")
        cer_prefix = _find_prefix("cer")
        wer_keys = [k for k in scores if wer_prefix and k.startswith(wer_prefix)]
        cer_keys = [k for k in scores if cer_prefix and k.startswith(cer_prefix)]
        main_keys = [k for k in scores if k not in wer_keys and k not in cer_keys]

        lines = [header, "-" * 40]
        for k in main_keys:
            lines.append(f"  {k:<25s} {scores[k]:.4f}")

        if wer_keys and wer_prefix:
            lines.append(f"  WER components (%) [{wer_prefix.rstrip('_')}]:")
            for k in wer_keys:
                lines.append(f"    {k.removeprefix(wer_prefix):<21s} {scores[k]:.1f}")
            total = sum(
                scores[f"{wer_prefix}{op}"]
                for op in ("delete", "insert", "replace", "equal")
            )
            err = total - scores.get(f"{wer_prefix}equal", 0.0)
            if total > 0:
                lines.append(f"    {'WER':<21s} {err / total * 100:.2f}%")

        if cer_keys and cer_prefix:
            lines.append(f"  CER components (%) [{cer_prefix.rstrip('_')}]:")
            for k in cer_keys:
                lines.append(f"    {k.removeprefix(cer_prefix):<21s} {scores[k]:.1f}")
            total = sum(
                scores[f"{cer_prefix}{op}"]
                for op in ("delete", "insert", "replace", "equal")
            )
            err = total - scores.get(f"{cer_prefix}equal", 0.0)
            if total > 0:
                lines.append(f"    {'CER':<21s} {err / total * 100:.2f}%")

        lines.append("-" * 40)
        logger.info("\n".join(lines))

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
