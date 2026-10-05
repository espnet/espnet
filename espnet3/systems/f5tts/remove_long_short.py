"""The ``remove_long_short`` stage: filter manifests by audio duration.

``F5TTSSystem.remove_long_short`` is a thin method that calls
:func:`remove_long_short` with the training config. The stage reads the TSV
manifests an F5-TTS recipe's ``create_dataset`` stage writes, one row per
utterance: ``utt_id``, ``wav_path``, ``text``, then any further columns.

Durations are read from audio headers in parallel through the
:class:`RemoveLongShortProvider` / :class:`RemoveLongShortRunner` pair.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import soundfile as sf
from omegaconf import DictConfig

from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import set_parallel

logger = logging.getLogger(__name__)


def _get_required_config(config, key: str, error_message: str):
    """Return ``config[key]``, raising ``RuntimeError`` when it is missing.

    Same contract as ``BaseSystem._get_required_config``, kept here so the
    stage function can be called without a system instance.
    """
    value = config.get(key, None) if config is not None else None
    if value is None:
        raise RuntimeError(error_message)
    return value


def load_manifest_entries(
    manifest_path: Union[str, Path],
) -> Tuple[List[Tuple[str, str, str]], int]:
    r"""Parse a TSV manifest into its usable rows.

    Each line is expected to be ``utt_id\twav_path\ttext\tspeaker_id``; any
    columns after the third are carried along untouched. Blank lines are
    skipped. Rows without text are dropped here, mirroring espnet2's
    ``NF != 1`` filter, so they never reach duration filtering.

    Args:
        manifest_path: Manifest written by the recipe's ``create_dataset``
            stage.

    Returns:
        A tuple ``(entries, num_dropped_empty)``. ``entries`` is a list of
        ``(utt_id, wav_path, line)`` tuples in manifest order, where ``line``
        is the original row, newline-terminated for direct write-back.
        ``num_dropped_empty`` counts the rows dropped for having no text.

    Raises:
        FileNotFoundError: If ``manifest_path`` does not exist.

    Examples:
        >>> entries, num_dropped_empty = load_manifest_entries(
        ...     "data/manifest/train.tsv"
        ... )
        >>> entries[0]
        ('103_1241_000000_000001', '/corpus/103/a.wav',
         '103_1241_000000_000001\t/corpus/103/a.wav\thello\t103\n')
    """
    entries: List[Tuple[str, str, str]] = []
    num_dropped_empty = 0
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            stripped_line = line.rstrip("\n")
            if not stripped_line:
                continue
            parts = stripped_line.split("\t")
            if len(parts) < 3 or parts[2].strip() == "":
                num_dropped_empty += 1
                continue
            utt_id, wav_path = parts[0], parts[1]
            entries.append(
                (utt_id, wav_path, line if line.endswith("\n") else line + "\n")
            )
    return entries, num_dropped_empty


class RemoveLongShortProvider(EnvironmentProvider):
    """Provider for the ``remove_long_short`` stage.

    Builds the manifest entries and duration bounds shared by
    :class:`RemoveLongShortRunner` workers. No model is loaded: duration
    filtering only reads audio headers.

    Args:
        config: Training config. Kept for the ``EnvironmentProvider``
            contract; the stage settings travel in ``params``.
        params: ``manifest_path``, ``min_duration`` and ``max_duration``,
            forwarded from the driver to every worker.

    Examples:
        >>> provider = RemoveLongShortProvider(
        ...     config=training_config,
        ...     params={
        ...         "manifest_path": "data/manifest/train.tsv",
        ...         "min_duration": 1.0,
        ...         "max_duration": 20.0,
        ...     },
        ... )
        >>> sorted(provider.build_env_local())
        ['entries', 'max_duration', 'min_duration', 'num_dropped_empty']
    """

    def __init__(self, config: DictConfig, params: Optional[Dict[str, Any]] = None):
        """Store the stage parameters handed to every worker."""
        super().__init__(config)
        self.params = params or {}

    def build_env_local(self) -> Dict[str, Any]:
        """Build the environment once on the driver for local execution.

        Returns:
            The manifest entries and duration bounds consumed by
            :meth:`RemoveLongShortRunner.forward`.

        Raises:
            RuntimeError: If ``manifest_path`` or a duration bound is missing
                from ``params``.
        """
        return RemoveLongShortProvider._build_env(self.params)

    def build_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Create the worker setup function for distributed execution.

        Returns:
            A zero-argument callable, run once per worker, that returns the
            same environment as :meth:`build_env_local`.
        """
        params = self.params

        def build_worker_env() -> Dict[str, Any]:
            return RemoveLongShortProvider._build_env(params)

        return build_worker_env

    @staticmethod
    def _build_env(params: Dict[str, Any]) -> Dict[str, Any]:
        manifest_path = params.get("manifest_path", None)
        if manifest_path is None:
            raise RuntimeError(
                "Please provide manifest_path obtained from create_dataset stage"
            )

        min_duration = params.get("min_duration", None)
        max_duration = params.get("max_duration", None)
        if min_duration is None or max_duration is None:
            raise RuntimeError(
                "min_duration and max_duration must be provided for "
                "remove_long_short stage."
            )

        entries, num_dropped_empty = load_manifest_entries(manifest_path)

        return {
            "entries": entries,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "num_dropped_empty": num_dropped_empty,
        }


class RemoveLongShortRunner(BaseRunner):
    """Runner that checks audio durations against the bounds in parallel.

    Each shard appends its status dicts to a shard-local ``results.jsonl``
    file, and :meth:`merge` reads every shard file back and re-sorts by
    ``idx``, so callers receive results in manifest order regardless of
    shard completion order.

    Examples:
        >>> runner = RemoveLongShortRunner(
        ...     provider=provider,
        ...     output_dir="data/manifest_filtered/shards",
        ...     shard_subdir="train",
        ...     resume=False,
        ... )
        >>> runner([0, 1])
        [{'idx': 0, 'utt_id': 'utt_a', 'keep': True},
         {'idx': 1, 'utt_id': 'utt_b', 'keep': False}]
    """

    @staticmethod
    def forward(
        idx: Union[int, Iterable[int]],
        entries: List[Tuple[str, str, str]],
        min_duration: float,
        max_duration: float,
        **env,
    ) -> Union[Dict[str, Any], list]:
        """Check the duration bounds for one index or a batch of indices.

        Args:
            idx: Single index or iterable of indices into ``entries``.
            entries: ``(utt_id, wav_path, line)`` tuples from the manifest.
            min_duration: Minimum allowed duration in seconds (exclusive).
            max_duration: Maximum allowed duration in seconds (exclusive).
            **env: Additional environment entries, unused.

        Returns:
            A status dict for an int index, or a list of status dicts for an
            iterable. Each entry is
            ``{"idx": int, "utt_id": str, "keep": bool}``.
        """
        if isinstance(idx, int):
            return RemoveLongShortRunner._process_one(
                idx, entries, min_duration, max_duration
            )
        return [
            RemoveLongShortRunner._process_one(
                one_idx, entries, min_duration, max_duration
            )
            for one_idx in idx
        ]

    @staticmethod
    def _process_one(
        idx: int,
        entries: List[Tuple[str, str, str]],
        min_duration: float,
        max_duration: float,
    ) -> Dict[str, Any]:
        utt_id, wav_path, _ = entries[idx]
        duration = sf.info(wav_path).duration

        # Strict inequalities to match espnet2 tts.sh awk filter.
        keep = not (duration <= min_duration or duration >= max_duration)
        return {"idx": idx, "utt_id": utt_id, "keep": keep}

    @staticmethod
    def open_writers(shard_dir: Optional[Path], **env) -> Dict[str, Any]:
        """Open the shard-local JSONL results file."""
        results_path = Path(shard_dir) / "results.jsonl"
        return {"results": results_path.open("w", encoding="utf-8")}

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
        **env,
    ) -> None:
        """Append one ``forward`` result (or batch of results) to the shard file."""
        records = result if isinstance(result, list) else [result]
        for record in records:
            writers["results"].write(json.dumps(record) + "\n")

    def merge(self, shard_dirs: List[Path]) -> List[Dict[str, Any]]:
        """Concatenate shard results and restore manifest (``idx``) order.

        Each shard's ``results.jsonl`` holds one JSON object per line, e.g.::

            {"idx": 0, "utt_id": "103_1241_000000_000001", "keep": true}
            {"idx": 1, "utt_id": "103_1241_000000_000002", "keep": false}
            {"idx": 2, "utt_id": "103_1241_000001_000000", "keep": true}
        """
        records: List[Dict[str, Any]] = []
        for shard_dir in shard_dirs:
            results_path = Path(shard_dir) / "results.jsonl"
            if not results_path.exists():
                continue
            with results_path.open("r", encoding="utf-8") as results_file:
                for line in results_file:
                    if line.strip():
                        records.append(json.loads(line))
        records.sort(key=lambda record: record["idx"])
        return records


def remove_long_short(config: DictConfig) -> None:
    """Write duration-filtered copies of the recipe's manifests.

    The body of the ``remove_long_short`` stage. For every configured split
    it reads the manifest written by ``create_dataset``, drops rows without
    text, checks each remaining utterance's duration from its audio header
    (in parallel, honouring ``config.parallel``), and writes the surviving
    rows, unchanged and in order, to ``save_path/<manifest file name>``.
    Re-running the stage overwrites the filtered manifests.

    Configuration should include (under ``remove_long_short``):

      - ``min_wav_duration``: Minimum duration in seconds. Utterances at or
        below it are dropped.
      - ``max_wav_duration``: Maximum duration in seconds. Utterances at or
        above it are dropped.
      - ``save_path``: Directory in which to save the filtered manifests.
      - ``splits``: Splits to process, a name or a list of names. Defaults
        to ``[train, valid, test]``.
      - ``manifest_paths``: Optional mapping from split to manifest path.
        A split without an entry uses ``data/manifest/{split}.tsv``,
        relative to the working directory.
      - ``batch_size``: Optional number of utterances per runner call.

    Args:
        config: Training config holding the ``remove_long_short`` block and,
            optionally, ``parallel``.

    Raises:
        RuntimeError: If the ``remove_long_short`` block, ``save_path`` or a
            duration bound is missing, or a split's manifest does not exist.

    Examples:
        .. code-block:: yaml

            remove_long_short:
              min_wav_duration: 1.0
              max_wav_duration: 20.0
              save_path: ${data_dir}/manifest_filtered
              splits: [train, valid, test]
              manifest_paths:
                train: ${data_dir}/manifest/train.tsv

        .. code-block:: python

            >>> remove_long_short(training_config)  # doctest: +SKIP
    """
    remove_long_short_config = _get_required_config(
        config,
        "remove_long_short",
        "training_config.remove_long_short must be set for remove_long_short stage.",
    )
    save_dir = Path(
        _get_required_config(
            remove_long_short_config,
            "save_path",
            "training_config.remove_long_short.save_path must be set "
            "for remove_long_short stage.",
        )
    )

    duration_error = (
        "training_config.remove_long_short.min_wav_duration and "
        "max_wav_duration must be set for remove_long_short stage."
    )
    min_duration = _get_required_config(
        remove_long_short_config, "min_wav_duration", duration_error
    )
    max_duration = _get_required_config(
        remove_long_short_config, "max_wav_duration", duration_error
    )

    # Set up parallelism before the duration-filtering runner is built.
    if config.get("parallel"):
        set_parallel(config.parallel)

    splits = remove_long_short_config.get("splits", ["train", "valid", "test"])
    if isinstance(splits, str):
        splits = [splits]

    manifest_paths = remove_long_short_config.get("manifest_paths", {})
    batch_size = remove_long_short_config.get("batch_size", None)

    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Removing long-short utterances with min_duration=%ss, max_duration=%ss",
        min_duration,
        max_duration,
    )

    for split in splits:
        logger.info("Processing split: %s", split)

        manifest_path = manifest_paths.get(split) if manifest_paths else None
        if manifest_path is None:
            manifest_path = f"data/manifest/{split}.tsv"
        manifest_path = Path(manifest_path).resolve()
        filtered_manifest_path = save_dir / manifest_path.name
        if not manifest_path.exists():
            raise RuntimeError(
                f"Manifest file not found for split '{split}': "
                f"{manifest_path}. Please generate the manifest file using "
                "the create_dataset stage and ensure the path is correct."
            )

        entries, num_dropped_empty = load_manifest_entries(manifest_path)
        num_entries = len(entries)

        provider = RemoveLongShortProvider(
            config=config,
            params={
                "manifest_path": str(manifest_path),
                "min_duration": min_duration,
                "max_duration": max_duration,
            },
        )

        # resume=False: the keep/drop decisions depend on the duration
        # bounds, so shard results from an earlier run (possibly with
        # different bounds) must never be reused.
        runner = RemoveLongShortRunner(
            provider=provider,
            batch_size=batch_size,
            output_dir=save_dir / "shards",
            shard_subdir=split,
            resume=False,
        )

        logger.info(
            "Checking durations for %d utterances (split: %s)", num_entries, split
        )

        # merge() returns the shard records flattened and re-sorted by idx.
        results = runner(list(range(num_entries))) if num_entries else []
        keep_by_idx = {record["idx"]: record["keep"] for record in results}

        filtered_lines = [
            line for idx, (_, _, line) in enumerate(entries) if keep_by_idx[idx]
        ]
        num_kept = len(filtered_lines)
        num_dropped_duration = num_entries - num_kept

        with open(filtered_manifest_path, "w", encoding="utf-8") as filtered_file:
            filtered_file.writelines(filtered_lines)

        logger.info(
            "Split '%s': kept %d, dropped %d by duration, dropped %d by empty "
            "text -> %s",
            split,
            num_kept,
            num_dropped_duration,
            num_dropped_empty,
            filtered_manifest_path,
        )

    logger.info(
        "Long-short utterance removal completed. Filtered manifests saved to: %s",
        save_dir,
    )
