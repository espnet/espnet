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
    """
    Entry point to collect feature statistics from a dataset.

    - Local single-process path
    - Dask parallel path
    - Multi-iterator (sharded) path (parallel only; no raw feats saved)
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
