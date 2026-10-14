"""VERSA-based metric for the F5-TTS `measure` stage.

Wraps `versa.bin.scorer` (https://github.com/wavlab-speech/versa) as an
ESPnet3 `BaseMetric`. The F5-TTS evaluation protocol (faster-whisper WER,
speaker similarity, UTMOS) is a VERSA score config, declared in
`egs3/TEMPLATE/f5tts/conf/metrics.yaml`; this class runs it over the
`infer` outputs of a test set and aggregates the per-utterance records.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set

import yaml
from omegaconf import OmegaConf

from espnet3.api import Field
from espnet3.components.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

# Lines the scorer prints when one metric fails while the run still exits 0.
_VERSA_FAILURE_MARKERS = ("Failed to load metric", "Error computing metric")


class VersaMetric(BaseMetric):
    """Score `infer` outputs by shelling out to `versa.bin.scorer`.

    The declaration follows the metric's own configuration, as
    :class:`~espnet3.components.metrics.base_metric.BaseMetric` allows:
    ``inputs`` are the synthesized wav (``wav_key``), the reference wav
    (``ref_key``) and, when ``text_key`` is set, the transcript, all read
    as files; ``outputs`` are the score names the caller promises, since
    VERSA's own keys exist only once it has run. The `measure` stage
    checks the result against them, so a configured metric that VERSA
    dropped fails the stage instead of vanishing from the table.

    VERSA scores files, so the state is the rows to score (``utt_id`` and
    one path or transcript per input): :meth:`update` adds a batch of
    them, numbered as they arrive, and :meth:`compute` writes them as SCP
    files, runs the scorer once over everything added, and returns the
    per-utterance average of every numeric field VERSA emitted, plus a
    corpus-level WER/CER pooled from the edit-operation counts when a
    recognition metric is configured. :meth:`__call__`, which `measure`
    uses once per test set, reads the rows from the `infer` stage's SCP
    files with their real utterance ids and scores them the same way
    under ``<output_dir>/<test_name>/scoring/versa_eval``. A metric that
    fails or yields no value raises instead of being dropped.

    Example:
        Declared from a recipe `conf/metrics.yaml`:

        .. code-block:: yaml

            metrics:
              - metric:
                  _target_: espnet3.systems.f5tts.metrics.versa.VersaMetric
                  score_config:
                    - name: signal_metric
                    - name: pseudo_mos
                      predictor_types: [utmos]
                  outputs: [mcd, utmos]
                  wav_key: wav
                  ref_key: ref
                  use_gpu: true
                inputs:
                  wav: wav
                  ref: dataset:ref_wav_path

        Or directly, pointing at an existing VERSA config file:

        .. code-block:: python

            metric = VersaMetric(score_config="conf/versa.yaml", outputs=["mcd"])
            scores = metric(
                {"wav": Path("exp/inference/test/wav.scp"),
                 "ref": Path("exp/inference/test/ref.scp")},
                test_name="test",
                output_dir=Path("exp/inference"),
            )

        Or in memory, from paths, as a callback would:

        .. code-block:: python

            metric.reset()
            metric.update(wav=["gen/a.wav"], ref=["prompt/a.wav"])
            metric.compute()
            # -> {"mcd": 3.1416, ...}
    """

    def __init__(
        self,
        score_config,
        outputs: Sequence[str],
        wav_key: str = "wav",
        ref_key: str = "ref",
        text_key: str | None = None,
        use_gpu: bool = True,
        io: str = "soundfile",
    ) -> None:
        """Store versa scorer settings and declare the contract they imply.

        Args:
            score_config: Path to a versa score-config YAML file, or an
                inline list of versa metric definitions.
            outputs: The score names this metric promises to return, such
                as ``["fwhisper_wer", "spk_similarity", "utmos"]``; each
                becomes a declared ``number`` output. VERSA returns more
                keys than these (the WER edit counts, say); extra keys are
                returned too, only the declared ones are checked.
            wav_key: Input alias for the synthesized-wav SCP file.
            ref_key: Input alias for the reference-wav SCP file (the prompt
                each utterance was synthesized from, for speaker similarity).
            text_key: Optional input alias for the transcript SCP file the
                recognition metric scores against. ``None`` omits ``--text``
                from the scorer command, for a score config without one;
                a name declares a required text input.
            use_gpu: Pass ``--use_gpu`` to the scorer.
            io: Value for the scorer's ``--io`` option.

        Raises:
            TypeError: If ``outputs`` is empty, or an alias repeats.

        Example:
            .. code-block:: python

                # Inline metric list; a versa_config.yaml is written at score time.
                metric = VersaMetric(
                    score_config=[{"name": "signal_metric"}], outputs=["mcd"]
                )

                # An existing versa config file, scored on CPU.
                metric = VersaMetric(
                    score_config="conf/versa.yaml", outputs=["mcd"], use_gpu=False
                )
        """
        self.score_config = score_config
        self.wav_key = wav_key
        self.ref_key = ref_key
        self.text_key = text_key
        self.use_gpu = use_gpu
        self.io = io
        inputs = [Field(wav_key, "audio"), Field(ref_key, "audio")]
        if text_key is not None:
            inputs.append(Field(text_key, "text"))
        self.inputs = tuple(inputs)
        self.outputs = tuple(Field(str(name), "number") for name in outputs)
        super().__init__()
        self.add_state("rows", [], "cat")
        # Where compute() scores: set by __call__ for a test set, else a
        # temporary directory.
        self._eval_dir: Optional[Path] = None

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
    def _as_file_entry(name: str, value: Any) -> str:
        """Return ``value`` as the SCP entry VERSA reads for input ``name``.

        VERSA opens files: an audio input must already be a path, since
        writing arrays out is not this metric's job, and a text input is
        its transcript.

        Raises:
            TypeError: If an audio input is not a path.
        """
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str):
            return value
        raise TypeError(
            f"VersaMetric input {name!r} must be a path (VERSA scores files), "
            f"not {type(value).__name__}"
        )

    def update(self, **values: Sequence[Any]) -> None:
        """Add one batch of rows to score, numbered in arrival order.

        Args:
            **values: One sequence per declared input name: paths for the
                wav inputs, transcripts for the text input.

        Raises:
            TypeError: If an audio value is not a path.

        Examples:
            >>> metric.update(wav=["gen/a.wav"], ref=["prompt/a.wav"])
        """
        names = [f.name for f in self.inputs if values.get(f.name) is not None]
        if not names:
            return
        for i in range(len(values[names[0]])):
            self.rows.append(
                {
                    "utt_id": f"utt{len(self.rows) + 1:08d}",
                    **{
                        name: self._as_file_entry(name, values[name][i])
                        for name in names
                    },
                }
            )

    def compute(self) -> Dict[str, float]:
        """Score every row added so far with one VERSA run.

        The rows are written as SCP files, one per input, under the test
        set's ``scoring/versa_eval`` directory when called from
        :meth:`__call__`, else under a temporary directory; the averages
        are also written there as ``avg_result.json``.

        Returns:
            Dict of metric name to per-utterance average, plus the pooled
            ``<prefix>_wer`` / ``<prefix>_cer`` percentage when the scorer
            emitted edit-operation counts (see :meth:`_aggregate`).

        Raises:
            ValueError: If no row was added.
            FileNotFoundError: If ``score_config`` is a path that does not
                exist.
            subprocess.CalledProcessError: If the VERSA scorer exits non-zero.
            RuntimeError: If the scorer exited 0 but a configured metric
                failed to load or computed no value for any utterance.

        Examples:
            >>> metric.update(wav=["gen/a.wav"], ref=["prompt/a.wav"])
            >>> metric.compute()
            {'mcd': 3.1416}
        """
        if not self.rows:
            raise ValueError("nothing to score: no rows were added")
        eval_dir = self._eval_dir or Path(tempfile.mkdtemp(prefix="versa_eval_"))
        eval_dir.mkdir(parents=True, exist_ok=True)

        scp_files: Dict[str, Path] = {}
        for name in (f.name for f in self.inputs if f.name in self.rows[0]):
            scp_files[name] = eval_dir / f"{name}.scp"
            with scp_files[name].open("w", encoding="utf-8") as f:
                for row in self.rows:
                    f.write(f"{row['utt_id']} {row[name]}\n")

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
            str(scp_files[self.wav_key]),
            "--gt",
            str(scp_files[self.ref_key]),
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
            cmd.extend(["--text", str(scp_files[self.text_key])])
        if self.use_gpu:
            cmd.append("--use_gpu")

        logger.info("Running VERSA: %s", " ".join(cmd))
        failures = self._run_scorer(cmd)

        averages = self._aggregate(result_file)
        self._reject_partial_results(averages, result_file, failures)
        average_path = eval_dir / "avg_result.json"
        with average_path.open("w") as f:
            json.dump(averages, f, indent=2)
        logger.info(
            "Wrote VERSA averages to %s (%d metrics)", average_path, len(averages)
        )
        return averages

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Score one test set with versa and return averaged metrics.

        The rows are read from the `infer` stage's SCP files with their own
        utterance ids, so ``result.json`` is keyed as the test set is; this
        is why the files are read here rather than through :meth:`update`,
        which numbers rows itself.

        Args:
            data: Mapping from input alias to the SCP file path written by
                the infer stage.
            test_name: Name of the test set being scored.
            output_dir: Root inference output directory; the scorer's files
                go under ``<output_dir>/<test_name>/scoring/versa_eval``.
                ``None`` scores under a temporary directory.

        Returns:
            What :meth:`compute` returns.

        Raises:
            KeyError: If ``wav_key``, ``ref_key``, or a configured
                ``text_key`` is absent from *data*.
            Whatever :meth:`compute` raises.

        Example:
            .. code-block:: python

                metric = VersaMetric(
                    score_config=[{"name": "signal_metric"}], outputs=["mcd"]
                )
                scores = metric(
                    {"wav": output_dir / "test" / "wav.scp",
                     "ref": output_dir / "test" / "ref.scp"},
                    test_name="test",
                    output_dir=output_dir,
                )
                # -> {"mcd": 3.1416, "sdr": 12.7, ...}
        """
        for key in (f.name for f in self.inputs):
            if key not in data:
                raise KeyError(
                    f"VersaMetric requires '{key}' input. Got: {list(data.keys())}"
                )

        self.reset()
        names = [f.name for f in self.inputs]
        for utt_id, row in self.iter_inputs(data, *names):
            self.rows.append({"utt_id": utt_id, **row})
        self._eval_dir = (
            Path(output_dir) / test_name / "scoring" / "versa_eval"
            if output_dir is not None
            else None
        )
        try:
            averages = self.compute()
        finally:
            self._eval_dir = None
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
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with process.stdout:
            for line in process.stdout:
                sys.stdout.write(line)
                if any(marker in line for marker in _VERSA_FAILURE_MARKERS):
                    failures.append(line.strip())
        sys.stdout.flush()
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        return failures

    @staticmethod
    def _iter_records(result_file: Path) -> Iterator[dict]:
        """Yield the per-utterance JSON records of the scorer's output file.

        Blank lines and the non-JSON trailer line VERSA may write are
        skipped, so every reader of ``result.json`` sees the same records.
        """
        with result_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record

    @staticmethod
    def _find_null_only_keys(result_file: Path) -> Set[str]:
        """Return keys that were null somewhere and never numeric anywhere.

        A metric that loads but throws per utterance leaves its key present
        and ``null`` in every record. Keys that are always text (``ref_text``,
        ``fwhisper_hyp_text``) are not implicated, because they are never null.
        """
        nulled: Set[str] = set()
        numeric: Set[str] = set()
        for record in VersaMetric._iter_records(result_file):
            for key, value in record.items():
                if value is None:
                    nulled.add(key)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
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
        percentage. Called automatically at the end of ``__call__``.

        Args:
            scores: Mapping of metric name to value, as returned by
                ``__call__``.
            test_name: Test-set name shown in the header. Omit for a
                generic header.

        Returns:
            None. The summary is emitted through this module's logger at
            INFO level.

        Example:
            .. code-block:: python

                VersaMetric.summarize({"mcd": 3.14, "sdr": 12.7}, "test")

            logs:

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
            reference_length = sum(
                scores[f"{prefix}{op}"] for op in ("delete", "replace", "equal")
            )
            errors = sum(
                scores[f"{prefix}{op}"] for op in ("delete", "replace", "insert")
            )
            if reference_length > 0:
                lines.append(f"    {label:<21s} {errors / reference_length * 100:.2f}%")

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
        for record in VersaMetric._iter_records(result_file):
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    sums[key] = sums.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1
        averages = {key: round(sums[key] / counts[key], 4) for key in sums}

        for metric in ("wer", "cer"):
            prefix = VersaMetric._find_prefix(sums, metric)
            if prefix is None:
                continue
            reference_length = sum(
                sums[f"{prefix}{op}"] for op in ("delete", "replace", "equal")
            )
            if reference_length <= 0:
                # Empty reference across the whole corpus: the rate is
                # undefined, so emit nothing rather than divide by zero.
                continue
            errors = sum(
                sums[f"{prefix}{op}"] for op in ("delete", "replace", "insert")
            )
            averages[prefix.rstrip("_")] = round(errors / reference_length * 100, 4)

        return averages
