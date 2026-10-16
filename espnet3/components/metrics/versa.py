"""VERSA-based metric wrapper for the `measure` stage.

Wraps `versa.bin.scorer` (https://github.com/wavlab-speech/versa) as an
ESPnet3 `BaseMetric`, so any generation task (codec resynthesis, TTS,
speech enhancement) can score its `infer` outputs from a recipe config
without a task-specific wrapper.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import yaml
from omegaconf import OmegaConf

from espnet3.api import Field
from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

# Lines the scorer prints when one metric fails while the run still exits 0.
_VERSA_FAILURE_MARKERS = ("Failed to load metric", "Error computing metric")


class VersaMetric(BaseMetric):
    """Score `infer` outputs by shelling out to `versa.bin.scorer`.

    Declares, as the metrics contract asks, what it reads and returns:
    ``hyp`` (the model's audio, required), ``ref`` (the reference audio,
    optional, since not every VERSA measure needs one) and ``text`` (a
    transcript, optional); and one ``number`` output per name in
    ``outputs``. VERSA decides the keys it emits from its own measures, so
    the config names the ones it relies on; the metric raises when any of
    them is missing from the scorer's result, and returns every numeric
    key the scorer emitted (plus a corpus-level WER/CER pooled from the
    edit-operation counts when a recognition measure is configured), so
    extra measures are not lost.

    `update` collects the paths of each batch; `compute` writes them as SCP
    files, runs the scorer once over all of them, and returns the
    per-utterance averages. A metric that fails or yields no value raises
    instead of being dropped from the result. The scorer's own files
    (``versa_config.yaml``, ``result.json``, ``avg_result.json``) are kept
    under ``<test_dir>/scoring/versa_eval`` when `measure` runs it.

    Example:
        Declared from a recipe `conf/metrics.yaml`:

        .. code-block:: yaml

            metrics:
              - metric:
                  _target_: espnet3.components.metrics.versa.VersaMetric
                  score_config:
                    - name: signal_metric
                    - name: pseudo_mos
                      predictor_types: [utmos]
                  outputs: [sdr, si_snr, utmos]
                  use_gpu: true
                inputs:
                  hyp: resyn_audio      # the model's audio output
                  ref: dataset:wav_path # the test set's reference waveform

        Or directly, pointing at an existing VERSA config file:

        .. code-block:: python

            >>> metric = VersaMetric(score_config="conf/versa.yaml", outputs=["mcd"])
            >>> scores = metric(
            ...     {"hyp": Path("exp/inference/test/resyn_audio.scp"),
            ...      "ref": Path("exp/inference/test/dataset/wav_path.scp")},
            ...     test_name="test",
            ...     output_dir=Path("exp/inference"),
            ... )
    """

    def __init__(
        self,
        score_config: Union[str, Path, Sequence[Dict[str, Any]]],
        outputs: Sequence[str],
        use_gpu: bool = True,
        io: str = "soundfile",
    ) -> None:
        """Declare the contract and store the scorer settings.

        Args:
            score_config: Path to a versa score-config YAML file, or an
                inline list of versa metric definitions.
            outputs: The result keys the scorer must emit for this
                configuration (e.g. ``[sdr, si_snr, utmos]``); each becomes
                a declared ``number`` output.
            use_gpu: Pass ``--use_gpu`` to the scorer.
            io: Value for the scorer's ``--io`` option.

        Raises:
            ValueError: If ``outputs`` is empty.

        Example:
            An inline metric list; ``versa_config.yaml`` is written at score
            time:

            .. code-block:: python

                >>> metric = VersaMetric(
                ...     score_config=[{"name": "signal_metric"}], outputs=["sdr"]
                ... )

            An existing VERSA config file, scored on the CPU:

            .. code-block:: python

                >>> metric = VersaMetric(
                ...     score_config="conf/versa.yaml", outputs=["mcd"], use_gpu=False
                ... )
        """
        if not outputs:
            raise ValueError(
                "VersaMetric needs `outputs`: the result keys the configured "
                "VERSA measures emit (e.g. [sdr, si_snr, utmos])."
            )
        self.score_config = score_config
        self.use_gpu = use_gpu
        self.io = io
        self.inputs = (
            Field("hyp", "audio", "Hypothesis audio"),
            Field("ref", "audio", "Reference audio", optional=True),
            Field("text", "text", "Transcript", optional=True),
        )
        self.outputs = tuple(Field(str(name), "number") for name in outputs)
        super().__init__()
        self.add_state("hyp_paths", [], "cat")
        self.add_state("ref_paths", [], "cat")
        self.add_state("texts", [], "cat")
        self._eval_dir: Optional[Path] = None

    def reset(self) -> None:
        """Clear the collected paths and forget the last scorer directory.

        Example:
            .. code-block:: python

                >>> metric.reset()
        """
        super().reset()
        self._eval_dir = None

    def update(
        self,
        *,
        hyp: Sequence[str],
        ref: Optional[Sequence[str]] = None,
        text: Optional[Sequence[str]] = None,
    ) -> None:
        """Collect one batch of file paths (and transcripts) to score.

        Args:
            hyp: Paths of the model's audio, one per utterance.
            ref: Paths of the reference audio, aligned with ``hyp``.
            text: Transcripts, aligned with ``hyp``, for recognition measures.

        Example:
            .. code-block:: python

                >>> metric.update(hyp=["a.wav"], ref=["a_ref.wav"])
        """
        self.hyp_paths.extend(str(p) for p in hyp)
        if ref is not None:
            self.ref_paths.extend(str(p) for p in ref)
        if text is not None:
            self.texts.extend(str(t) for t in text)

    def compute(self) -> Dict[str, float]:
        """Run the scorer over the collected files and return the averages.

        The scorer runs in a fresh temporary directory (``write_artifacts``
        moves its files next to the test set afterwards); the utterances are
        numbered in the order they were collected, since the contract hands
        ``update`` values only.

        Returns:
            Dict of result key to per-utterance average, plus the pooled
            ``<prefix>_wer`` / ``<prefix>_cer`` percentage when the scorer
            emitted edit-operation counts (see :meth:`_aggregate`). Every
            declared output is present.

        Raises:
            ValueError: If nothing was collected.
            FileNotFoundError: If ``score_config`` is a path that does not
                exist.
            subprocess.CalledProcessError: If the VERSA scorer exits non-zero.
            RuntimeError: If the scorer exited 0 but a measure failed to load,
                computed no value for any utterance, or a declared output is
                missing from the result.

        Example:
            .. code-block:: python

                >>> metric.update(hyp=["a.wav"], ref=["a_ref.wav"])
                >>> metric.compute()
                {'sdr': 12.7, 'si_snr': 10.1}
        """
        if not self.hyp_paths:
            raise ValueError("nothing to score: no hypothesis audio was collected")
        if self.ref_paths and len(self.ref_paths) != len(self.hyp_paths):
            raise ValueError(
                f"{len(self.ref_paths)} reference paths for "
                f"{len(self.hyp_paths)} hypothesis paths"
            )
        eval_dir = Path(tempfile.mkdtemp(prefix="versa_eval_"))
        self._eval_dir = eval_dir
        utt_ids = [f"utt{i:08d}" for i in range(len(self.hyp_paths))]
        pred_scp = self._write_scp(eval_dir / "pred.scp", utt_ids, self.hyp_paths)
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
            str(pred_scp),
            "--score_config",
            str(score_config_path),
            "--cache_folder",
            str(eval_dir / "cache"),
            "--output_file",
            str(result_file),
            "--io",
            self.io,
        ]
        if self.ref_paths:
            gt_scp = self._write_scp(eval_dir / "gt.scp", utt_ids, self.ref_paths)
            cmd.extend(["--gt", str(gt_scp)])
        if self.texts:
            text_scp = self._write_scp(eval_dir / "text", utt_ids, self.texts)
            cmd.extend(["--text", str(text_scp)])
        if self.use_gpu:
            cmd.append("--use_gpu")

        logger.info("Running VERSA: %s", " ".join(cmd))
        failures = self._run_scorer(cmd)
        averages = self._aggregate(result_file)
        self._reject_partial_results(averages, result_file, failures)
        missing = [f.name for f in self.outputs if f.name not in averages]
        if missing:
            raise RuntimeError(
                f"VERSA emitted no value for declared outputs {missing}; "
                f"it emitted {sorted(averages)}. Check `outputs` against the "
                "measures in `score_config`."
            )
        with (eval_dir / "avg_result.json").open("w") as f:
            json.dump(averages, f, indent=2)
        self.summarize(averages)
        return averages

    def write_artifacts(self, test_dir: Path) -> None:
        """Keep the scorer's files under ``<test_dir>/scoring/versa_eval``.

        Moves ``versa_config.yaml``, ``result.json`` (one record per
        utterance) and ``avg_result.json`` out of the temporary directory
        :meth:`compute` ran in, and discards the rest of it.

        Args:
            test_dir: The test set's output directory, already created.

        Example:
            .. code-block:: python

                >>> metric.write_artifacts(Path("exp/inference/test"))
        """
        if self._eval_dir is None:
            return
        target = Path(test_dir) / "scoring" / "versa_eval"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("versa_config.yaml", "result.json", "avg_result.json"):
            source = self._eval_dir / name
            if source.is_file():
                shutil.move(str(source), str(target / name))
        shutil.rmtree(self._eval_dir, ignore_errors=True)
        self._eval_dir = None
        logger.info("Wrote VERSA results to %s", target)

    @staticmethod
    def _write_scp(path: Path, utt_ids: Sequence[str], values: Sequence[str]) -> Path:
        """Write ``<utt_id> <value>`` lines and return ``path``."""
        with path.open("w", encoding="utf-8") as f:
            for utt_id, value in zip(utt_ids, values):
                f.write(f"{utt_id} {value}\n")
        return path

    def _resolve_score_config_path(self, eval_dir: Path) -> Path:
        """Return a YAML file path VERSA can `open()`.

        If ``score_config`` is a path to an existing file, return it directly.
        Otherwise treat it as an inline config object and dump it to
        ``eval_dir/versa_config.yaml``.
        """
        if isinstance(self.score_config, (str, Path)):
            config_path = Path(self.score_config)
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"VERSA score_config path does not exist: {config_path}"
                )
            logger.info("Using VERSA config file %s", config_path)
            return config_path

        score_config = self.score_config
        if OmegaConf.is_config(score_config):
            # Hydra instantiate passes inline lists/dicts as OmegaConf
            # containers, which yaml.safe_dump cannot represent.
            score_config = OmegaConf.to_container(score_config, resolve=True)

        config_path = eval_dir / "versa_config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(score_config, f, sort_keys=False)
        logger.info("Wrote inline VERSA metric list to %s", config_path)
        return config_path

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
    def _find_null_only_keys(result_file: Path) -> Set[str]:
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

        null_only = VersaMetric._find_null_only_keys(result_file)
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
        percentage. :meth:`compute` calls it on the aggregated scores.

        Args:
            scores: Mapping of metric name to value, as returned by
                :meth:`compute`.
            test_name: Test-set name shown in the header. Omit for a
                generic header.

        Returns:
            None. The summary is emitted through this module's logger at
            INFO level.

        Example:
            .. code-block:: python

                >>> VersaMetric.summarize({"mcd": 3.14, "sdr": 12.7}, "test")

            logs the table:

            .. code-block:: text

                VERSA scores - test
                ----------------------------------------
                  mcd                       3.1400
                  sdr                       12.7000
                ----------------------------------------
        """
        header = f"VERSA scores - {test_name}" if test_name else "VERSA scores"

        wer_prefix = VersaMetric._find_prefix(scores, "wer")
        cer_prefix = VersaMetric._find_prefix(scores, "cer")
        wer_keys = [key for key in scores if wer_prefix and key.startswith(wer_prefix)]
        cer_keys = [key for key in scores if cer_prefix and key.startswith(cer_prefix)]

        # The pooled rates _aggregate adds are printed in their own section.
        pooled_keys = {
            prefix.rstrip("_") for prefix in (wer_prefix, cer_prefix) if prefix
        }
        main_keys = [
            key
            for key in scores
            if key not in wer_keys and key not in cer_keys and key not in pooled_keys
        ]

        lines = [header, "-" * 40]
        for key in main_keys:
            lines.append(f"  {key:<25s} {scores[key]:.4f}")

        for label, prefix, keys in (
            ("WER", wer_prefix, wer_keys),
            ("CER", cer_prefix, cer_keys),
        ):
            if not (keys and prefix):
                continue
            lines.append(f"  {label} components (%) [{prefix.rstrip('_')}]:")
            for key in keys:
                lines.append(f"    {key.removeprefix(prefix):<21s} {scores[key]:.1f}")
            # Reference length is delete + replace + equal: insertions are
            # errors but not reference tokens, so they never enter the
            # denominator (VERSA asserts the same identity itself).
            ref_len = sum(
                scores[f"{prefix}{op}"] for op in ("delete", "replace", "equal")
            )
            errors = sum(
                scores[f"{prefix}{op}"] for op in ("delete", "replace", "insert")
            )
            if ref_len > 0:
                lines.append(f"    {label:<21s} {errors / ref_len * 100:.2f}%")

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

        Example:
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
