# espnet3/collect_stats/collect_stats.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from espnet3.utils.collect_stats_local import collect_stats_local
from espnet3.utils.collect_stats_parallel import (
    collect_stats_multiple_iterator,
    collect_stats_parallel,
)


def collect_stats(
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    output_dir: Path,
    task: Optional[str] = None,
    parallel_config: Optional[dict] = None,
    write_collected_feats: bool = True,
    batch_size: int = 4,
):
    """Entry point for collecting dataset statistics used for feature normalization.

    Depending on the supplied configuration the function chooses one of three
    execution strategies:

    - Local single-process execution.
    - Parallel execution using Dask.
    - Multi-iterator (sharded) execution, which only works with parallel
      processing and does not write raw features to disk.

    Args:
        model_config: Configuration object used to instantiate the model that
            extracts features from the input examples.
        dataset_config: Configuration of the dataset organizer providing the
            split specified by ``mode``.
        dataloader_config: Dataloader configuration. The attribute matching
            ``mode`` may include the ``multiple_iterator`` flag.
        mode: Name of the dataset split to process (``train`` or ``valid``).
        output_dir: Directory where aggregated statistics and optionally
            collected features are written.
        task: Name of the ESPnet task. If ``None``, ``model_config`` should be
            directly instantiable.
        parallel_config: Configuration for parallel execution. Required when
            ``multiple_iterator`` is enabled.
        write_collected_feats: Whether to persist the raw collected features.
            This option is unsupported in multi-iterator mode.
        batch_size: Number of dataset items processed per batch.

    Raises:
        RuntimeError: If ``multiple_iterator`` is ``True`` but
            ``parallel_config`` is not provided.
        ValueError: If ``write_collected_feats`` is ``True`` when running in
            multi-iterator mode.

    Returns:
        None: Aggregated statistics are saved under ``output_dir / mode``.
    """
    mode_config = getattr(dataloader_config, mode)
    if getattr(mode_config, "multiple_iterator", False):
        if parallel_config is None:
            raise RuntimeError("You should set parallel config with multiple iterator.")
        if write_collected_feats:
            raise ValueError(
                "Currently this option is not supported in multi-iterator mode."
            )
        sum_dict, sq_dict, count_dict = collect_stats_multiple_iterator(
            model_config,
            dataset_config,
            dataloader_config,
            mode,
            output_dir,
            task,
            batch_size,
            parallel_config,
        )

    elif parallel_config is None:
        sum_dict, sq_dict, count_dict = collect_stats_local(
            model_config,
            dataset_config,
            dataloader_config,
            mode,
            output_dir,
            task,
            write_collected_feats,
            batch_size,
        )
    else:
        sum_dict, sq_dict, count_dict = collect_stats_parallel(
            model_config,
            dataset_config,
            dataloader_config,
            mode,
            output_dir,
            task,
            write_collected_feats,
            batch_size,
            parallel_config,
        )

    # Persist aggregated stats (same as before)
    for key in sum_dict:
        (output_dir / mode).mkdir(parents=True, exist_ok=True)
        np.savez(
            output_dir / mode / f"{key}_stats.npz",
            count=count_dict[key],
            sum=sum_dict[key],
            sum_square=sq_dict[key],
        )
    with open(output_dir / mode / "stats_keys", "w") as f:
        f.write("\n".join(sum_dict) + "\n")
