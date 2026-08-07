"""VERSA-based TTS metric wrapper for the measure stage."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

import yaml

from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

_VERSA_FAILURE_MARKERS = ("Failed to load metric", "Error computing metric")


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
            sys.executable,
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
        failures = self._run_scorer(cmd)

        averages = self._aggregate(result_file)
        self._reject_partial_results(averages, result_file, failures)
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
    def _run_scorer(cmd: List[str]) -> List[str]:
        """Run the scorer, echoing its output, and return its failure lines.

        Output is streamed straight through to stdout so the log looks exactly
        as it would if VERSA had inherited the terminal, while each line is
        also inspected for the failure markers above.
        """
        failures: List[str] = []
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                if any(marker in line for marker in _VERSA_FAILURE_MARKERS):
                    failures.append(line.strip())
        sys.stdout.flush()
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        return failures

    @staticmethod
    def _null_only_keys(result_file: Path) -> Set[str]:
        """Return keys that were null somewhere and never numeric anywhere.

        A metric that loads but throws per utterance leaves its key present and
        ``null`` in every record. Keys that are always text (``ref_text``,
        ``fwhisper_hyp_text``) are not implicated, because they are never null.
        """
        nulled: Set[str] = set()
        numeric: Set[str] = set()
        with result_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # VERSA writes a non-JSON trailer line
                if not isinstance(record, dict):
                    continue
                for key, value in record.items():
                    if value is None:
                        nulled.add(key)
                    elif isinstance(value, (int, float)) and not isinstance(
                        value, bool
                    ):
                        numeric.add(key)
        return nulled - numeric

    @staticmethod
    def _reject_partial_results(
        averages: Dict[str, float],
        result_file: Path,
        failures: List[str],
    ) -> None:
        """Raise unless every configured metric actually produced a score.

        VERSA exits 0 when an individual metric fails, so without this the
        stage would report whichever metrics happened to survive and look
        entirely successful. A silently truncated score set is worse than a
        failed run: the numbers reach a table with nothing marking them as
        incomplete.
        """
        problems = list(failures)

        null_only = VersaMetric._null_only_keys(result_file)
        if null_only:
            problems.append(
                "computed no value for any utterance: " + ", ".join(sorted(null_only))
            )
        if not averages:
            problems.append(f"no numeric scores at all in {result_file}")

        if problems:
            raise RuntimeError(
                "VERSA did not produce a complete score set.\n  "
                + "\n  ".join(problems)
                + f"\nScores that did succeed: {sorted(averages) or 'none'}."
                "\nThe measure stage needs versa, faster-whisper,"
                " openai-whisper (for the whisper_basic text cleaner) and"
                " s3prl (for the speaker model's WavLM front-end); a missing"
                " one disables its metric without failing the run."
            )

    @staticmethod
    def _find_prefix(scores: Dict[str, float], metric: str) -> str | None:
        """Return ``'<prefix>_<metric>_'`` when all four edit ops are present.

        VERSA emits WER/CER as four per-utterance counts (delete, insert,
        replace, equal) under a backend-specific prefix, e.g.
        ``fwhisper_wer_delete``. Returns ``None`` when the group is absent or
        incomplete.
        """
        for key in scores:
            if key.endswith(f"_{metric}_delete"):
                prefix = key[: -(len(metric) + 8)]
                ops = [
                    f"{prefix}_{metric}_{op}"
                    for op in ("delete", "insert", "replace", "equal")
                ]
                if all(op in scores for op in ops):
                    return f"{prefix}_{metric}_"
        return None

    @staticmethod
    def summarize(scores: Dict[str, float], test_name: str = "") -> None:
        """Log a formatted summary of VERSA scores."""
        header = f"VERSA scores - {test_name}" if test_name else "VERSA scores"

        wer_prefix = VersaMetric._find_prefix(scores, "wer")
        cer_prefix = VersaMetric._find_prefix(scores, "cer")
        wer_keys = [k for k in scores if wer_prefix and k.startswith(wer_prefix)]
        cer_keys = [k for k in scores if cer_prefix and k.startswith(cer_prefix)]

        pooled_keys = {
            prefix.rstrip("_") for prefix in (wer_prefix, cer_prefix) if prefix
        }
        main_keys = [
            k
            for k in scores
            if k not in wer_keys and k not in cer_keys and k not in pooled_keys
        ]

        lines = [header, "-" * 40]
        for k in main_keys:
            lines.append(f"  {k:<25s} {scores[k]:.4f}")

        if wer_keys and wer_prefix:
            lines.append(f"  WER components (%) [{wer_prefix.rstrip('_')}]:")
            for k in wer_keys:
                lines.append(f"    {k.removeprefix(wer_prefix):<21s} {scores[k]:.1f}")

            ref_len = sum(
                scores[f"{wer_prefix}{op}"] for op in ("delete", "replace", "equal")
            )
            err = sum(
                scores[f"{wer_prefix}{op}"] for op in ("delete", "replace", "insert")
            )
            if ref_len > 0:
                lines.append(f"    {'WER':<21s} {err / ref_len * 100:.2f}%")

        if cer_keys and cer_prefix:
            lines.append(f"  CER components (%) [{cer_prefix.rstrip('_')}]:")
            for k in cer_keys:
                lines.append(f"    {k.removeprefix(cer_prefix):<21s} {scores[k]:.1f}")

            ref_len = sum(
                scores[f"{cer_prefix}{op}"] for op in ("delete", "replace", "equal")
            )
            err = sum(
                scores[f"{cer_prefix}{op}"] for op in ("delete", "replace", "insert")
            )
            if ref_len > 0:
                lines.append(f"    {'CER':<21s} {err / ref_len * 100:.2f}%")

        lines.append("-" * 40)
        logger.info("\n".join(lines))

    @staticmethod
    def _aggregate(result_file: Path) -> Dict[str, float]:
        """Aggregate per-utterance VERSA records into corpus-level scores.

        Most keys are returned as the plain per-utterance mean. WER and CER
        are the exception: they are returned as a pooled PERCENTAGE under
        ``<prefix><metric>`` (e.g. ``fwhisper_wer`` -> ``3.45`` meaning
        3.45%), computed from the pooled edit-op counts rather than the mean
        of per-utterance rates.

        The pooled rate is ``(delete + replace + insert) / (delete + replace +
        equal) * 100``. The denominator is the REFERENCE length, which does
        not include insertions; insertions are errors but are not reference
        tokens. This matches VERSA's own definition, which asserts
        ``delete + replace + equal == len(ref_words)`` in
        ``versa/corpus_metrics/fwhisper_wer.py``. A rate above 100% is
        therefore possible and correct when insertions dominate.
        """
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
        averages = {key: round(sums[key] / counts[key], 4) for key in sums}

        # Corpus-level WER/CER: pool the raw counts across every utterance,
        # then divide once. Averaging per-utterance rates instead would let
        # short utterances dominate. The denominator is the pooled REFERENCE
        # length (delete + replace + equal), never the alignment length, so
        # insertions raise the numerator without inflating the denominator.
        # Emitted as a PERCENTAGE, to match how summarize() prints it
        # (e.g. 3.45 means 3.45%).
        for metric in ("wer", "cer"):
            prefix = VersaMetric._find_prefix(sums, metric)
            if prefix is None:
                continue
            ref_len = sum(
                sums[f"{prefix}{op}"] for op in ("delete", "replace", "equal")
            )
            if ref_len <= 0:
                # Empty reference across the whole corpus: the rate is
                # undefined, so emit nothing rather than divide by zero.
                continue
            errors = sum(
                sums[f"{prefix}{op}"] for op in ("delete", "replace", "insert")
            )
            averages[prefix.rstrip("_")] = round(errors / ref_len * 100, 4)

        return averages
