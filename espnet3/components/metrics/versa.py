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
import sys
from pathlib import Path
from typing import Dict, List, Set

import yaml
from omegaconf import OmegaConf

from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

# Lines the scorer prints when one metric fails while the run still exits 0.
_VERSA_FAILURE_MARKERS = ("Failed to load metric", "Error computing metric")


class VersaMetric(BaseMetric):
    """Score `infer` outputs by shelling out to `versa.bin.scorer`.

    One instance scores one test set: `__call__` receives the SCP files the
    `infer` stage wrote, runs the VERSA scorer over them as a subprocess, and
    returns the per-utterance average of every numeric field VERSA emitted,
    plus a corpus-level WER/CER pooled from the edit-operation counts when a
    recognition metric is configured. A metric that fails or yields no value
    raises instead of being dropped from the result.

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
            output_dir=Path("exp/inference"),
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
        output_dir: Path,
    ) -> Dict[str, float]:
        """Score one test set with versa and return averaged metrics.

        Args:
            data: Mapping from input alias to the SCP file path written by
                the infer stage.
            test_name: Name of the test set being scored.
            output_dir: Root inference output directory.

        Returns:
            Dict of metric name to per-utterance average, plus the pooled
            ``<prefix>_wer`` / ``<prefix>_cer`` percentage when the scorer
            emitted edit-operation counts (see :meth:`_aggregate`). The same
            values are written to
            ``<output_dir>/<test_name>/scoring/versa_eval/avg_result.json``.

        Raises:
            KeyError: If ``wav_key``, ``ref_key``, or a configured
                ``text_key`` is absent from *data*.
            FileNotFoundError: If ``score_config`` is a path that does not
                exist.
            subprocess.CalledProcessError: If the VERSA scorer exits non-zero.
            RuntimeError: If the scorer exited 0 but a configured metric
                failed to load or computed no value for any utterance.

        Examples:
            ```python
            metric = VersaMetric(score_config=[{"name": "signal_metric"}])
            scores = metric(
                {"wav": output_dir / "test" / "wav.scp",
                 "ref": output_dir / "test" / "ref.scp"},
                test_name="test",
                output_dir=output_dir,
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

        eval_dir = Path(output_dir) / test_name / "scoring" / "versa_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        score_config_path = self._resolve_score_config_path(eval_dir)
        result_file = eval_dir / "result.json"

        # The interpreter running this stage, not whatever `python` resolves to
        # on PATH: under srun or a venv those differ, and only this one is known
        # to have versa installed.
        cmd = [
            sys.executable,
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
        also inspected for the markers in ``_VERSA_FAILURE_MARKERS``.

        Args:
            cmd: The scorer command line.

        Returns:
            The output lines that reported a failed metric, in order.

        Raises:
            subprocess.CalledProcessError: If the scorer exits non-zero.
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

        A metric that loads but throws per utterance leaves its key present
        and ``null`` in every record. Keys that are always text (``ref_text``,
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

        Raises:
            RuntimeError: Naming each failed metric and each key that never
                received a value, and listing the scores that did succeed.
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
                "\nA metric whose dependency is not installed is disabled by"
                " VERSA without failing the run; install it and rerun measure."
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

        wer_prefix = VersaMetric._find_prefix(scores, "wer")
        cer_prefix = VersaMetric._find_prefix(scores, "cer")
        wer_keys = [k for k in scores if wer_prefix and k.startswith(wer_prefix)]
        cer_keys = [k for k in scores if cer_prefix and k.startswith(cer_prefix)]

        # The pooled rates _aggregate adds are printed in their own section.
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

        for label, prefix, keys in (
            ("WER", wer_prefix, wer_keys),
            ("CER", cer_prefix, cer_keys),
        ):
            if not (keys and prefix):
                continue
            lines.append(f"  {label} components (%) [{prefix.rstrip('_')}]:")
            for k in keys:
                lines.append(f"    {k.removeprefix(prefix):<21s} {scores[k]:.1f}")
            # Reference length is delete + replace + equal: insertions are
            # errors but not reference tokens, so they never enter the
            # denominator (VERSA asserts the same identity itself).
            ref_len = sum(
                scores[f"{prefix}{op}"] for op in ("delete", "replace", "equal")
            )
            err = sum(scores[f"{prefix}{op}"] for op in ("delete", "replace", "insert"))
            if ref_len > 0:
                lines.append(f"    {label:<21s} {err / ref_len * 100:.2f}%")

        lines.append("-" * 40)
        logger.info("\n".join(lines))

    @staticmethod
    def _aggregate(result_file: Path) -> Dict[str, float]:
        """Aggregate per-utterance VERSA records into corpus-level scores.

        Most keys are returned as the plain per-utterance mean. WER and CER
        are the exception: they are returned as a pooled PERCENTAGE under
        ``<prefix>_<metric>`` (e.g. ``fwhisper_wer`` -> ``3.45`` meaning
        3.45%), computed from the pooled edit-operation counts rather than
        the mean of per-utterance rates, which would let short utterances
        dominate.

        The pooled rate is ``(delete + replace + insert) / (delete + replace +
        equal) * 100``. The denominator is the REFERENCE length, which does
        not include insertions; insertions are errors but are not reference
        tokens. This matches VERSA's own definition, which asserts
        ``delete + replace + equal == len(ref_words)`` in
        ``versa/corpus_metrics/fwhisper_wer.py``. A rate above 100% is
        therefore possible and correct when insertions dominate.

        Args:
            result_file: The scorer's JSONL output, one record per utterance.

        Returns:
            Mapping of metric name to its corpus-level value, rounded to four
            decimals. Non-numeric and boolean fields are ignored.

        Examples:
            Two utterances with ``fwhisper_wer_{delete,insert,replace,equal}``
            counts ``(0, 1, 1, 8)`` and ``(1, 0, 0, 9)`` give
            ``fwhisper_wer == 15.7895``: 3 errors over a reference length of
            19, not the mean of the two per-utterance rates.
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
