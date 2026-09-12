"""Classification system implementation.

This module adds classification-specific stages on top of the base system:
duration-based utterance filtering and label list creation.
"""

import logging
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig

from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.remove_long_short_provider import RemoveLongShortProvider
from espnet3.systems.base.remove_long_short_runner import RemoveLongShortRunner
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class CLSSystem(BaseSystem):
    """Classification-specific system.

    This system adds:
      - Removing utterances outside a duration range
      - Building the label list consumed as ``token_list``

    Additional stage log paths:
        | Stage             | Path reference                              |
        |---                |---                                          |
        | remove_long_short | training_config.remove_long_short.save_path |
        | prepare_labels    | training_config.prepare_labels.save_path    |
    """

    def __init__(
        self,
        training_config: DictConfig | None = None,
        inference_config: DictConfig | None = None,
        metrics_config: DictConfig | None = None,
        publication_config: DictConfig | None = None,
        stage_log_mapping: dict | None = None,
        demo_config: DictConfig | None = None,
    ) -> None:
        """Initialize the classification system with optional stage configs.

        Args:
            training_config: Training configuration.
            inference_config: Inference configuration.
            metrics_config: Measurement configuration.
            publication_config: Publication configuration for model packing
                and upload stages.
            stage_log_mapping: Optional per-stage log directory overrides.
            demo_config: Demo configuration for demo packing and upload
                stages.
        """
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            publication_config=publication_config,
            stage_log_mapping={
                "remove_long_short": "training_config.remove_long_short.save_path",
                "prepare_labels": "training_config.prepare_labels.save_path",
                **(stage_log_mapping or {}),
            },
            demo_config=demo_config,
        )

    def remove_long_short(self, *args, **kwargs):
        """Remove utterances outside the configured duration range.

        This stage reads WAV headers via soundfile (in parallel, through
        ``RemoveLongShortProvider`` / ``RemoveLongShortRunner``) and writes
        filtered manifests for downstream stages. It mirrors stage 3 of the
        ESPnet2 ``cls.sh`` recipe.

        Configuration should include (under
        ``training_config.remove_long_short``):
            - ``min_wav_duration``: Minimum duration in seconds
            - ``max_wav_duration``: Maximum duration in seconds
            - ``save_path``: Directory to save filtered manifests
            - ``splits``: List of splits to process (default:
              ``[train, valid, test]``)
            - ``manifest_paths``: Optional dict of split to manifest path
              (default: ``data/manifest/{split}.tsv``)

        Example:
            .. code-block:: yaml

                remove_long_short:
                  min_wav_duration: 0.1
                  max_wav_duration: 20
                  save_path: data/manifest_filtered
                  splits: [train, valid]

        Raises:
            RuntimeError: If required configuration is missing or a manifest
                file is not found.
        """
        self._reject_stage_args("remove_long_short", args, kwargs)
        logger.info(
            "CLSSystem.remove_long_short(): starting long-short utterance removal"
        )

        remove_long_short_config = self._get_required_config(
            self.training_config,
            "remove_long_short",
            "training_config.remove_long_short must be set for "
            "remove_long_short stage.",
        )
        save_path = Path(
            self._get_required_config(
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
        min_duration = self._get_required_config(
            remove_long_short_config, "min_wav_duration", duration_error
        )
        max_duration = self._get_required_config(
            remove_long_short_config, "max_wav_duration", duration_error
        )

        # Parse the parallel configuration early to set up parallelism for
        # the duration-filtering runner.
        if self.training_config.get("parallel"):
            set_parallel(self.training_config.parallel)

        splits = remove_long_short_config.get("splits", ["train", "valid", "test"])
        if isinstance(splits, str):
            splits = [splits]

        manifest_paths = remove_long_short_config.get("manifest_paths", {})
        batch_size = remove_long_short_config.get("batch_size", None)

        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Removing long-short utterances with "
            f"min_duration={min_duration}s, max_duration={max_duration}s"
        )

        for split in splits:
            logger.info(f"Processing split: {split}")

            manifest_path = manifest_paths.get(split) if manifest_paths else None
            if manifest_path is None:
                manifest_path = f"data/manifest/{split}.tsv"
            manifest_path = Path(manifest_path).resolve()
            filtered_manifest_path = save_path / manifest_path.name
            if not manifest_path.exists():
                raise RuntimeError(
                    f"Manifest file not found for split '{split}': "
                    f"{manifest_path}. Please generate the manifest file using "
                    "the create_dataset stage and ensure the path is correct."
                )

            entries, n_dropped_empty = RemoveLongShortProvider._load_entries(
                manifest_path
            )
            n_entries = len(entries)

            provider = RemoveLongShortProvider(
                config=self.training_config,
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
                output_dir=save_path / "shards",
                shard_subdir=split,
                resume=False,
            )

            logger.info(
                f"Checking durations for {n_entries} utterances (split: {split})"
            )

            indices = list(range(n_entries))
            # merge() returns the shard records flattened and re-sorted by idx.
            results = runner(indices) if n_entries else []
            keep_by_idx = {r["idx"]: r["keep"] for r in results}

            n_kept = 0
            n_dropped_duration = 0
            filtered_entries = []
            for idx, (_, _, line) in enumerate(entries):
                if keep_by_idx[idx]:
                    filtered_entries.append(line)
                    n_kept += 1
                else:
                    n_dropped_duration += 1

            with open(filtered_manifest_path, "w", encoding="utf-8") as f:
                f.writelines(filtered_entries)

            logger.info(
                f"Split '{split}': kept {n_kept}, dropped {n_dropped_duration} "
                f"by duration, dropped {n_dropped_empty} by empty text -> "
                f"{filtered_manifest_path}"
            )

        logger.info(
            "Long-short utterance removal completed. Filtered manifests "
            f"saved to: {save_path}"
        )

    def prepare_labels(self, *args, **kwargs):
        """Build the label list used as ``token_list`` by the classifier.

        Labels are collected from the third column of the training manifest,
        split on whitespace so that multi-label utterances contribute every
        label they carry. The list is ordered by descending frequency, which
        matches stage 4 of the ESPnet2 ``cls.sh`` recipe.

        Configuration should include (under
        ``training_config.prepare_labels``):
            - ``save_path``: Directory to write the label list into
            - ``filename``: Label list file name (e.g. ``token_list``)
            - ``manifest_path``: Training manifest to read
              (default: ``data/manifest/train.tsv``)
            - ``add_symbol``: Optional ``"<symbol>:<index>"`` entries; a
              negative index counts from the end

        Example:
            .. code-block:: yaml

                prepare_labels:
                  save_path: data
                  filename: token_list
                  manifest_path: data/manifest/train.tsv
                  add_symbol:
                    - "<unk>:-1"

        Raises:
            RuntimeError: If required configuration is missing, the manifest
                is not found, an ``add_symbol`` entry is malformed, or the
                manifest holds no label.
        """
        self._reject_stage_args("prepare_labels", args, kwargs)
        logger.info("CLSSystem.prepare_labels(): starting label list creation")

        prepare_labels_config = self._get_required_config(
            self.training_config,
            "prepare_labels",
            "training_config.prepare_labels must be set for prepare_labels stage.",
        )
        save_path_str = self._get_required_config(
            prepare_labels_config,
            "save_path",
            "training_config.prepare_labels.save_path must be set for "
            "prepare_labels stage.",
        )
        filename = self._get_required_config(
            prepare_labels_config,
            "filename",
            "training_config.prepare_labels.filename must be set "
            "(e.g. 'token_list'); save_path is the output directory.",
        )
        save_dir = Path(save_path_str)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / filename

        manifest_path = Path(
            prepare_labels_config.get("manifest_path", "data/manifest/train.tsv")
        ).resolve()
        if not manifest_path.exists():
            raise RuntimeError(
                f"Manifest file not found for label list creation: "
                f"{manifest_path}. Please ensure the manifest file is "
                "generated and the path is correct."
            )

        counter = Counter()
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                for label in parts[2].split():
                    counter[label] += 1

        if not counter:
            raise RuntimeError(f"No label found in manifest: {manifest_path}")

        # Sort by the number of occurrences in descending order.
        labels = [label for label, _ in sorted(counter.items(), key=lambda x: -x[1])]

        add_symbol = [str(item) for item in prepare_labels_config.get("add_symbol", [])]
        for symbol_and_id in add_symbol:
            # e.g. symbol="<unk>:-1"
            try:
                symbol, idx = symbol_and_id.split(":")
                idx = int(idx)
            except ValueError:
                raise RuntimeError(f"Format error: e.g. '<unk>:-1': {symbol_and_id}")
            symbol = symbol.strip()

            # e.g. idx=0  -> insert as the first symbol
            # e.g. idx=-1 -> append as the last symbol
            if idx < 0:
                idx = len(labels) + 1 + idx
            labels.insert(idx, symbol)

        with open(output_file, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(f"{label}\n")

        logger.info(
            "prepare_labels: wrote %d labels from %d occurrences -> %s",
            len(labels),
            sum(counter.values()),
            output_file,
        )
