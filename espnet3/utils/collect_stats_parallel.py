# espnet3/collect_stats/collect_parallel.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
from hydra.utils import instantiate
from tqdm import tqdm

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet3.parallel import get_client, parallel_for, set_parallel
from espnet3.task import get_espnet_model
from espnet3.utils.collect_stats_local import (
    _accumulate_and_persist_batch,
    _build_collate_fn,
    process_batch_batching,
)


def make_collect_setup_fn(
    *,
    task,
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    shard_idx: Optional[int] = None,
    write_collected_feats: bool = False,
):
    """
    Create setup_fn for Dask workers. Keys must match process_batch_batching().
    """

    def setup_fn():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = (
            get_espnet_model(task, model_config) if task else instantiate(model_config)
        )
        model = model.to(device).eval()

        organizer = instantiate(dataset_config)
        ds = getattr(organizer, mode)
        if shard_idx is not None:
            ds = ds.shard(shard_idx)

        collate_fn = _build_collate_fn(dataloader_config)

        return {
            "model": model,
            "dataset": ds,
            "collate_fn": collate_fn,
            "device": device,
            "write_collected_feats": write_collected_feats,
        }

    return setup_fn


def collect_stats_parallel(
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    output_dir: Path,
    task: Optional[str],
    write_collected_feats: bool,
    batch_size: int,
    parallel_config,
):
    """Collect feature statistics using Dask execution with setup_fn."""
    set_parallel(parallel_config)

    dummy = instantiate(dataset_config)
    num_items = len(getattr(dummy, mode))
    del dummy

    index_batches = [
        list(range(i, min(i + batch_size, num_items)))
        for i in range(0, num_items, batch_size)
    ]

    sum_dict, sq_dict, count_dict = (
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
    )
    writers = {}

    setup_fn = make_collect_setup_fn(
        task=task,
        model_config=model_config,
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        mode=mode,
        shard_idx=None,
        write_collected_feats=write_collected_feats,
    )

    with get_client() as client, DatadirWriter(output_dir / mode) as datadir_writer:
        for result in tqdm(
            parallel_for(
                process_batch_batching,
                index_batches,
                client=client,
                setup_fn=setup_fn,
            ),
            total=len(index_batches),
            desc=f"[{mode} parallel]",
        ):
            if write_collected_feats:
                stats, shape_info, feats = result
            else:
                stats, shape_info = result
                feats = None

            _accumulate_and_persist_batch(
                stats=stats,
                shape_info=shape_info,
                feats=feats,
                sum_dict=sum_dict,
                sq_dict=sq_dict,
                count_dict=count_dict,
                datadir_writer=datadir_writer,
                writers=writers,
                mode=mode,
                output_dir=output_dir,
                write_collected_feats=write_collected_feats,
            )

    return sum_dict, sq_dict, count_dict


def collect_stats_multiple_iterator(
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    output_dir: Path,
    task: Optional[str],
    batch_size: int,
    parallel_config,
):
    """
    Collect stats on sharded datasets using Dask + setup_fn.
    NOTE: raw features are NOT saved in multi-iterator mode.
    """
    set_parallel(parallel_config)

    sum_dict, sq_dict, count_dict = (
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
    )

    mode_cfg = getattr(dataloader_config, mode)
    num_shards = mode_cfg.num_shards

    for shard_idx in range(num_shards):
        dummy = instantiate(dataset_config)
        shard_len = len(getattr(dummy, mode).shard(shard_idx))
        del dummy

        batch_indices = [
            list(range(i, min(i + batch_size, shard_len)))
            for i in range(0, shard_len, batch_size)
        ]

        setup_fn = make_collect_setup_fn(
            task=task,
            model_config=model_config,
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
            mode=mode,
            shard_idx=shard_idx,
            write_collected_feats=False,
        )

        with get_client() as client, DatadirWriter(output_dir / mode) as datadir_writer:
            for shard_result in tqdm(
                parallel_for(
                    process_batch_batching,
                    batch_indices,
                    client=client,
                    setup_fn=setup_fn,
                ),
                total=len(batch_indices),
                desc=f"[{mode} parallel (Processing shard: {shard_idx}/{num_shards})]",
            ):
                shard_stats, shape_info = shard_result

                _accumulate_and_persist_batch(
                    stats=shard_stats,
                    shape_info=shape_info,
                    feats=None,
                    sum_dict=sum_dict,
                    sq_dict=sq_dict,
                    count_dict=count_dict,
                    datadir_writer=datadir_writer,
                    writers={},
                    mode=mode,
                    output_dir=output_dir,
                    write_collected_feats=False,
                    shape_key_suffix=f".shard.{shard_idx}",
                )

    return sum_dict, sq_dict, count_dict
