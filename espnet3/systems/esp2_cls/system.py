"""Classification system implementation.

This module adds classification-specific stages on top of the base system:
duration-based utterance filtering and label list creation.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import ClassVar

from espnet3.contract.stages import StageSpec
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.esp2_cls.src.remove_long_short_provider import (
    RemoveLongShortProvider,
)
from espnet3.systems.esp2_cls.src.remove_long_short_runner import RemoveLongShortRunner

logger = logging.getLogger(__name__)


class ESP2CLSSystem(BaseSystem):
    """Classification-specific system.

    This system adds ``remove_long_short``, which removes utterances outside
    a duration range, and ``prepare_labels``, which builds the label list
    consumed as ``token_list``. Both run between ``create_dataset`` and
    ``collect_stats``; their logs go under each stage's ``save_path``.

    Examples:
        >>> [s.name for s in ESP2CLSSystem.stages[:4]]
        ['create_dataset', 'remove_long_short', 'prepare_labels', 'collect_stats']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(
            name="remove_long_short",
            config="training",
            log_dir="remove_long_short.save_path",
        ),
        StageSpec(
            name="prepare_labels", config="training", log_dir="prepare_labels.save_path"
        ),
        StageSpec(name="collect_stats", config="training", log_dir="stats_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_model", config=("training", "publication")),
        StageSpec(name="upload_model", config="publication"),
        StageSpec(name="pack_demo", config="demo", log_dir="pack.out_dir"),
        StageSpec(name="upload_demo", config="demo", log_dir="pack.out_dir"),
    )

    def remove_long_short(self, *args, **kwargs):
        """Remove utterances outside the configured duration range.

        This stage reads WAV headers via soundfile (in parallel, through
        ``RemoveLongShortProvider`` / ``RemoveLongShortRunner``) and writes
        filtered manifests for downstream stages. It mirrors stage 3 of the
        ESPnet2 ``cls.sh`` recipe.

        Under ``training_config.remove_long_short``, configure:

        .. list-table::
           :header-rows: 1

           * - Field
             - Description
           * - ``min_wav_duration``
             - Minimum duration in seconds.
           * - ``max_wav_duration``
             - Maximum duration in seconds.
           * - ``save_path``
             - Directory in which to save filtered manifests.
           * - ``splits``
             - Splits to process. Defaults to ``[train, valid, test]``.
           * - ``manifest_paths``
             - Optional mapping from split to manifest path. Defaults to
               ``data/manifest/{split}.tsv``.

        Example:
            .. code-block:: yaml

                remove_long_short:
                  min_wav_duration: 0.1
                  max_wav_duration: 30
                  save_path: data/manifest_filtered
                  splits: [train, valid, test]

        Raises:
            RuntimeError: If required configuration is missing or a manifest
                file is not found.
        """
        self._reject_stage_args("remove_long_short", args, kwargs)
        logger.info(
            "ESP2CLSSystem.remove_long_short(): starting long-short utterance removal"
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

        Under ``training_config.prepare_labels``, configure:

        .. list-table::
           :header-rows: 1

           * - Field
             - Description
           * - ``save_path``
             - Directory to write the label list into.
           * - ``filename``
             - Label list file name, such as ``token_list``.
           * - ``manifest_path``
             - Training manifest to read. Defaults to
               ``data/manifest/train.tsv``.
           * - ``add_symbol``
             - Optional ``"<symbol>:<index>"`` entries; a negative index
               counts from the end.

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
        logger.info("ESP2CLSSystem.prepare_labels(): starting label list creation")

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
        n_unlabelled = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                labels = parts[2].split() if len(parts) >= 3 else []
                if not labels:
                    # The row still reaches training, so say so rather than
                    # letting it disappear from the label inventory in silence.
                    n_unlabelled += 1
                    continue
                for label in labels:
                    counter[label] += 1

        if n_unlabelled:
            logger.warning(
                "prepare_labels: %d row(s) in %s carry no label",
                n_unlabelled,
                manifest_path,
            )

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
