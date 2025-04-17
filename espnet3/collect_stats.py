import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import torch
from dask import delayed
from dask.distributed import get_worker, as_completed, WorkerPlugin
from hydra.utils import instantiate
from tqdm import tqdm

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet3 import get_espnet_model
from espnet3.parallel import set_parallel, get_client


class CollectStatsPlugin(WorkerPlugin):
    def __init__(self, task, model_config, dataset_config, dataloader_config, mode):
        self.task = task
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.mode = mode

    def setup(self, worker):
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
        if self.mode == "train":
            worker.dataset = organizer.train
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
    feats = {k: v.cpu().numpy() for k, v in feats.items()}

    uid_list = [uid for uid, _ in items]
    return uid_list, feats


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
    npy_scp_writers = {}
    if parallel_config is None:
        # ---- Local mode ----
        print(f"[{mode}] Running in local (non-parallel) mode")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_espnet_model(task, model_config) if task else instantiate(model_config)
        model = model.to(device).eval()

        organizer = instantiate(dataset_config)
        dataset = getattr(organizer, mode)
        if hasattr(dataloader_config, "collate_fn"):
            collate_fn = instantiate(dataloader_config.collate_fn)
        else:
            from espnet2.train.collate_fn import CommonCollateFn
            collate_fn = CommonCollateFn(int_pad_value=-1)

        sum_dict, sq_dict, count_dict, writers = defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0), {}
        with DatadirWriter(output_dir / mode) as datadir_writer:
            for i in tqdm(range(0, len(dataset), batch_size), desc=f"[{mode}]"):
                batch_idxs = list(range(i, min(i + batch_size, len(dataset))))
                items = [dataset[j] for j in batch_idxs]
                batch = collate_fn(items)
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch[1].items()}
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
                        datadir_writer[f"{key}_shape"][uid] = ",".join(map(str, seq.shape))
    else:
        # ---- Parallel mode ----
        set_parallel(parallel_config)
        plugin = CollectStatsPlugin(task, model_config, dataset_config, dataloader_config, mode)

        dummy = instantiate(dataset_config)
        dataset = getattr(dummy, mode)
        index_list = list(range(len(dataset)))
        del dummy  # free memory

        sum_dict, sq_dict, count_dict, writers = defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0), {}

        with get_client(plugin=plugin) as client, DatadirWriter(output_dir / mode) as datadir_writer:
            batch_idxs_list = [
                list(range(i, min(i + batch_size, len(dataset))))
                for i in range(0, len(dataset), batch_size)
            ]
            futures = client.map(process_batch_batching, batch_idxs_list)
            for future in tqdm(as_completed(futures), total=len(batch_idxs_list), desc=f"[{mode} parallel]"):
                uid_list, feats = future.result()
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
                            if (key, mode) not in writers:
                                p = output_dir / mode / "collect_feats"
                                writers[(key, mode)] = NpyScpWriter(p / f"data_{key}", p / f"{key}.scp")
                            writers[(key, mode)][uid] = seq
                        datadir_writer[f"{key}_shape"][uid] = ",".join(map(str, seq.shape))

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
