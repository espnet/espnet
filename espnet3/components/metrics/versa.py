"""VERSA-based metric wrapper for the `measure` stage.

Wraps `versa.bin.scorer` (https://github.com/wavlab-speech/versa) as an
ESPnet3 `BaseMetric`, so any generation task -- codec resynthesis, TTS,
speech enhancement -- can score its `infer` outputs from a recipe config
without a task-specific wrapper.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict

import yaml
from omegaconf import OmegaConf

from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


class VersaMetric(BaseMetric):
    """Score `infer` outputs by shelling out to `versa.bin.scorer`.

    One instance scores one test set: `__call__` receives the SCP files the
    `infer` stage wrote, runs the VERSA scorer over them as a subprocess, and
    returns the per-utterance average of every numeric field VERSA emitted.

    Examples:
        Declared from a recipe `conf/metrics.yaml`:
        ```yaml
        metrics:
          - metric:
              _target_: espnet3.components.metrics.versa.VersaMetric
              score_config:
                - name: signal_metric
                - name: pseudo_mos
                  predictor_types: [utmos]
              wav_key: wav
              ref_key: ref
              use_gpu: true
            inputs:
              wav: wav
              ref: ref
        ```

        Or directly, pointing at an existing VERSA config file:
        ```python
        metric = VersaMetric(score_config="conf/versa.yaml")
        scores = metric(
            {"wav": Path("exp/inference/test/wav.scp"),
             "ref": Path("exp/inference/test/ref.scp")},
            test_name="test",
            inference_dir=Path("exp/inference"),
        )
        ```
    """

    def __init__(
        self,
        score_config,
        wav_key: str = "wav",
        ref_key: str = "ref",
        text_key: str | None = None,
        use_gpu: bool = True,
        io: str = "soundfile",
    ) -> None:
        """Store versa scorer settings.

        Args:
            score_config: Path to a versa score-config YAML file, or an
                inline list of versa metric definitions.
            wav_key: Input alias for the resynthesized-wav SCP file.
            ref_key: Input alias for the ground-truth-wav SCP file.
            text_key: Optional input alias for a transcript SCP file. Codec
                reconstruction evaluation has no transcript, so this defaults
                to None and ``--text`` is omitted from the scorer command.
            use_gpu: Pass ``--use_gpu`` to the scorer.
            io: Value for the scorer's ``--io`` option.

        Examples:
            ```python
            # Inline metric list; a versa_config.yaml is written at score time.
            metric = VersaMetric(score_config=[{"name": "signal_metric"}])

            # An existing versa config file, scored on CPU.
            metric = VersaMetric(score_config="conf/versa.yaml", use_gpu=False)
            ```
        """
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

        score_config = self.score_config
        if OmegaConf.is_config(score_config):
            # Hydra instantiate passes inline lists/dicts as OmegaConf
            # containers, which yaml.safe_dump cannot represent.
            score_config = OmegaConf.to_container(score_config, resolve=True)

        out = eval_dir / "versa_config.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(score_config, f, sort_keys=False)
        logger.info("Wrote inline VERSA metric list to %s", out)
        return out

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Score one test set with versa and return averaged metrics.

        Args:
            data: Mapping from input alias to the SCP file path written by
                the infer stage.
            test_name: Name of the test set being scored.
            inference_dir: Root inference output directory.

        Returns:
            Dict of metric name to per-utterance average. The same values are
            written to ``<inference_dir>/<test_name>/scoring/versa_eval/
            avg_result.json``.

        Raises:
            KeyError: If ``wav_key``, ``ref_key``, or a configured
                ``text_key`` is absent from *data*.
            FileNotFoundError: If ``score_config`` is a path that does not
                exist.
            subprocess.CalledProcessError: If the VERSA scorer exits non-zero.

        Examples:
            ```python
            metric = VersaMetric(score_config=[{"name": "signal_metric"}])
            scores = metric(
                {"wav": inference_dir / "test" / "wav.scp",
                 "ref": inference_dir / "test" / "ref.scp"},
                test_name="test",
                inference_dir=inference_dir,
            )
            # -> {"mcd": 3.1416, "sdr": 12.7, ...}
            ```
        """
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
        if self.text_key is not None and self.text_key not in data:
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
            "--score_config",
            str(score_config_path),
            "--cache_folder",
            str(eval_dir / "cache"),
            "--output_file",
            str(result_file),
            "--io",
            self.io,
        ]
        if self.text_key is not None:
            cmd.extend(["--text", str(data[self.text_key])])
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
        """Log a formatted summary table of VERSA scores.

        Metrics are logged one per line. WER/CER edit-operation counts are
        detected by their ``<prefix>_{wer,cer}_{delete,insert,replace,equal}``
        naming, grouped into their own section, and reduced to a single
        percentage. Called automatically at the end of ``__call__``.

        Args:
            scores: Mapping of metric name to value, as returned by
                ``__call__``.
            test_name: Test-set name shown in the header. Omit for a
                generic header.

        Returns:
            None. The summary is emitted through this module's logger at
            INFO level.

        Examples:
            ```python
            VersaMetric.summarize({"mcd": 3.14, "sdr": 12.7}, "test")
            ```
            logs:
            ```text
            VERSA scores - test
            ----------------------------------------
              mcd                       3.1400
              sdr                       12.7000
            ----------------------------------------
            ```
        """
        header = f"VERSA scores - {test_name}" if test_name else "VERSA scores"

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
