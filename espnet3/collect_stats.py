import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from dask import delayed
from dask.distributed import WorkerPlugin, as_completed, get_worker
from hydra.utils import instantiate
from tqdm import tqdm

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet3 import get_espnet_model
from espnet3.parallel import get_client, set_parallel


class CollectStatsPlugin(WorkerPlugin):
    def __init__(self, task, model_config, dataset_config, dataloader_config,
                 mode, shard_idx=None, write_collected_feats=False):
        self.task = task
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.mode = mode
        self.shard_idx = shard_idx
        self.write_collected_feats = write_collected_feats

    def setup(self, worker):
        worker.write_collected_feats = self.write_collected_feats
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model
        if self.task:
            worker.model = get_espnet_model(self.task, self.model_config)
        else:
            worker.model = instantiate(self.model_config)
        worker.model.to(device).eval()
        worker.device = device

        # dataset
        organizer = instantiate(self.dataset_config)
        if self.mode == "train" and self.shard_idx is None:
            worker.dataset = organizer.train
        elif  self.mode == "train" and self.shard_idx is not None:
            worker.dataset = organizer.train.shard(self.shard_idx)
        elif self.mode == "valid":
            worker.dataset = organizer.valid
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # collate_fn
        if hasattr(self.dataloader_config, "collate_fn"):
            worker.collate_fn = instantiate(self.dataloader_config.collate_fn)
        else:
            from espnet2.train.collate_fn import CommonCollateFn

            worker.collate_fn = CommonCollateFn(int_pad_value=-1)


def process_batch_batching(idxs: list[int]):
    worker = get_worker()
    model = worker.model
    dataset = worker.dataset
    collate_fn = worker.collate_fn
    device = worker.device

    items = [dataset[i] for i in idxs]  # list of (uid, sample_dict)
    batch = collate_fn(items)
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch[1].items()}
    feats = model.collect_feats(**batch)
    feats = {k: v.detach().cpu().numpy() for k, v in feats.items()}

    stats = defaultdict(lambda: {"sum": 0, "sq": 0, "count": 0})
    shape_info = defaultdict(dict)
    
    uid_list = [uid for uid, _ in items]
    for idx, uid in enumerate(uid_list):
        for key in feats:
            seq = feats[key][idx]
            if f"{key}_lengths" in feats:
                length = feats[f"{key}_lengths"][idx]
                seq = seq[:length]
            else:
                seq = seq[None]

            stats[key]["sum"] += seq.sum(0)
            stats[key]["sq"] += (seq**2).sum(0)
            stats[key]["count"] += len(seq)

            shape_info[key][uid] = ",".join(map(str, seq.shape))

    if worker.write_collected_feats:
        return stats, shape_info, feats
    else:
        return stats, shape_info


def collect_stats_local(
    model_config,
    dataset_config,
    dataloader_config,
    mode,
    output_dir,
    task,
    write_collected_feats,
    batch_size,
):
    npy_scp_writers = {}
    # ---- Local mode ----
    print(f"[{mode}] Running in local (non-parallel) mode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        get_espnet_model(task, model_config) if task else instantiate(model_config)
    )
    model = model.to(device).eval()

    organizer = instantiate(dataset_config)
    dataset = getattr(organizer, mode)
    if hasattr(dataloader_config, "collate_fn"):
        collate_fn = instantiate(dataloader_config.collate_fn)
    else:
        from espnet2.train.collate_fn import CommonCollateFn

        collate_fn = CommonCollateFn(int_pad_value=-1)

    sum_dict, sq_dict, count_dict, writers = (
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        {},
    )
    with DatadirWriter(output_dir / mode) as datadir_writer:
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"[{mode}]"):
            batch_idxs = list(range(i, min(i + batch_size, len(dataset))))
            items = [dataset[j] for j in batch_idxs]
            batch = collate_fn(items)
            batch = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in batch[1].items()
            }
            feats = model.collect_feats(**batch)
            feats = {k: v.cpu().numpy() for k, v in feats.items()}
            uid_list = [uid for uid, _ in items]

            for idx, uid in enumerate(uid_list):
                for key in feats:
                    seq = feats[key][idx]
                    if f"{key}_lengths" in feats:
                        length = feats[f"{key}_lengths"][idx]
                        seq = seq[:length]
                    else:
                        seq = seq[None]

                    sum_dict[key] += seq.sum(0)
                    sq_dict[key] += (seq**2).sum(0)
                    count_dict[key] += len(seq)

                    if write_collected_feats:
                        if (key, mode) not in npy_scp_writers:
                            p = output_dir / mode / "collect_feats"
                            npy_scp_writers[(key, mode)] = NpyScpWriter(
                                p / f"data_{key}", p / f"{key}.scp"
                            )
                        npy_scp_writers[(key, mode)][uid] = seq
                    datadir_writer[f"{key}_shape"][uid] = ",".join(
                        map(str, seq.shape)
                    )
    return sum_dict, sq_dict, count_dict
    

def collect_stats_parallel(
    model_config,
    dataset_config,
    dataloader_config,
    mode,
    output_dir,
    task,
    write_collected_feats,
    batch_size,
    parallel_config,
):
    # ---- Parallel mode ----
    set_parallel(parallel_config)
    plugin = CollectStatsPlugin(
        task, model_config, dataset_config, dataloader_config, mode,
        write_collected_feats=write_collected_feats,
    )

    dummy = instantiate(dataset_config)
    dataset = getattr(dummy, mode)
    index_list = list(range(len(dataset)))
    del dummy  # free memory

    sum_dict, sq_dict, count_dict, writers = (
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        {},
    )

    with (
        get_client(plugin=plugin) as client,
        DatadirWriter(output_dir / mode) as datadir_writer,
    ):
        batch_idxs_list = [
            list(range(i, min(i + batch_size, len(dataset))))
            for i in range(0, len(dataset), batch_size)
        ]
        futures = client.map(process_batch_batching, batch_idxs_list)
        for future in tqdm(
            as_completed(futures),
            total=len(batch_idxs_list),
            desc=f"[{mode} parallel]",
        ):
            if write_collected_feats:
                stats, shape_info, feats = future.result()
            else:
                stats, shape_info = future.result()

            for key in stats:
                sum_dict[key] += stats[key]["sum"]
                sq_dict[key] += stats[key]["sq"]
                count_dict[key] += stats[key]["count"]
            
                for uid, shape_str in shape_info[key].items():
                    datadir_writer[f"{key}_shape"][uid] = shape_str

                if write_collected_feats:
                    for idx, uid in enumerate(shape_info[key].keys()):
                        for key in feats:
                            seq = feats[key][idx]
                            if (key, mode) not in writers:
                                p = output_dir / mode / "collect_feats"
                                writers[(key, mode)] = NpyScpWriter(
                                    p / f"data_{key}", p / f"{key}.scp"
                                )
                            writers[(key, mode)][uid] = seq

    return sum_dict, sq_dict, count_dict


def collect_stats_multiple_iterator(
    model_config,
    dataset_config,
    dataloader_config,
    mode,
    output_dir,
    task,
    batch_size,
    parallel_config,
):
    # ---- Parallel mode ----
    set_parallel(parallel_config)
    sum_dict, sq_dict, count_dict, writers = (
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        {},
    )

    # Process all shards
    dataloader_mode_config = getattr(dataloader_config, mode)
    num_shards = dataloader_mode_config.num_shards

    for shard_idx in range(num_shards):
        plugin = CollectStatsPlugin(
            task, model_config, dataset_config, dataloader_config, mode, shard_idx
        )

        dummy = instantiate(dataset_config)
        dataset = getattr(dummy, mode).shard(shard_idx)
        del dummy  # free memory

        with (
            get_client(plugin=plugin) as client,
            DatadirWriter(output_dir / mode) as datadir_writer,
        ):
            batch_idxs_list = [
                list(range(i, min(i + batch_size, len(dataset))))
                for i in range(0, len(dataset), batch_size)
            ]
            futures = client.map(process_batch_batching, batch_idxs_list)
            for future in tqdm(
                as_completed(futures),
                total=len(batch_idxs_list),
                desc=f"[{mode} parallel (Processing shard: {shard_idx}/{num_shards})]",
            ):
                shard_stats, shape_info = future.result()
                for key in shard_stats:
                    sum_dict[key] += shard_stats[key]["sum"]
                    sq_dict[key] += shard_stats[key]["sq"]
                    count_dict[key] += shard_stats[key]["count"]
                
                for uid, shape_str in shape_info[key].items():
                    datadir_writer[f"{key}_shape.shard.{shard_idx}"][uid] = shape_str
                
    return sum_dict, sq_dict, count_dict


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
    """Distributed or local stat collection via index-based processing."""
    # First check if we use multiple_iterator
    if getattr(dataloader_config, "multiple_iterator", True):
        if parallel_config is None:
            raise RuntimeError("You should set parallel config with multiple iterator.")
        if write_collected_feats:
            raise ValueError("Currently this option is not supported." \
                "If you really want to save all feats at this stage," \
                "comment out this warning and add feats as return value.")
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
        # ---- Local mode ----
        sum_dict, sq_dict, count_dict = collect_stats_local(
            model_config,
            dataset_config,
            dataloader_config,
            mode,
            output_dir,
            task,
            write_collected_feats,
            batch_size
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

    # Save statistics
    for key in sum_dict:
        np.savez(
            output_dir / mode / f"{key}_stats.npz",
            count=count_dict[key],
            sum=sum_dict[key],
            sum_square=sq_dict[key],
        )
    with open(output_dir / mode / "stats_keys", "w") as f:
        f.write("\n".join(sum_dict) + "\n")
