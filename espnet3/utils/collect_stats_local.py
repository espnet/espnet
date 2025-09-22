from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet3.task import get_espnet_model


def process_batch_batching(
    idxs: List[int],
    *,
    model=None,
    dataset=None,
    collate_fn=None,
    device: Optional[torch.device] = None,
    write_collected_feats: bool = False,
):
    """
    Process a batch of dataset indices to compute feature statistics.

    Returns:
        If write_collected_feats:
            Tuple[stats, shape_info, feats]
        else:
            Tuple[stats, shape_info]
    """
    items = [dataset[i] for i in idxs]  # list of (uid, sample_dict)
    uids, _ = zip(*items)
    batch = collate_fn(items)

    tensors = {
        k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch[1].items()
    }

    with torch.no_grad():
        feats = model.collect_feats(**tensors)

    feats = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
        for k, v in feats.items()
    }

    stats = defaultdict(lambda: {"sum": 0, "sq": 0, "count": 0})
    shape_info = defaultdict(dict)

    uid_list = list(uids)
    for b_idx, uid in enumerate(uid_list):
        for feat_key in list(feats.keys()):
            if f"{feat_key}_lengths" in feats:
                length = int(feats[f"{feat_key}_lengths"][b_idx])
                seq = feats[feat_key][b_idx][:length]
            else:
                seq = feats[feat_key][b_idx][None]

            stats[feat_key]["sum"] += seq.sum(0)
            stats[feat_key]["sq"] += (seq**2).sum(0)
            stats[feat_key]["count"] += len(seq)
            shape_info[feat_key][uid] = ",".join(map(str, seq.shape))

    if write_collected_feats:
        return stats, shape_info, feats
    else:
        return stats, shape_info


def _accumulate_and_persist_batch(
    *,
    stats: Dict,
    shape_info: Dict,
    feats: Optional[Dict],
    sum_dict: Dict,
    sq_dict: Dict,
    count_dict: Dict,
    datadir_writer: DatadirWriter,
    writers: Dict,
    mode: str,
    output_dir: Path,
    write_collected_feats: bool,
    shape_key_suffix: str = "",
):
    """Merge per-batch stats and persist shapes/features."""
    for feat_key, agg in stats.items():
        sum_dict[feat_key] += agg["sum"]
        sq_dict[feat_key] += agg["sq"]
        count_dict[feat_key] += agg["count"]

    for feat_key, uid2shape in shape_info.items():
        shape_key = f"{feat_key}_shape{shape_key_suffix}"
        for uid, shape_str in uid2shape.items():
            datadir_writer[shape_key][uid] = shape_str

        if write_collected_feats and feats is not None and feat_key in feats:
            uids_in_order = list(uid2shape.keys())
            feat_batch = feats[feat_key]
            len_key = f"{feat_key}_lengths"
            len_batch = feats.get(len_key, None)

            writer_key = (feat_key, mode)
            if writer_key not in writers:
                p = output_dir / mode / "collect_feats"
                writers[writer_key] = NpyScpWriter(
                    p / f"data_{feat_key}", p / f"{feat_key}.scp"
                )
            w = writers[writer_key]

            for b_idx, uid in enumerate(uids_in_order):
                seq = feat_batch[b_idx]
                if len_batch is not None:
                    L = int(len_batch[b_idx])
                    seq = seq[:L]
                else:
                    seq = seq[None]

                if not isinstance(seq, np.ndarray):
                    seq = np.asarray(seq)
                w[uid] = seq


def _build_collate_fn(dataloader_config):
    if hasattr(dataloader_config, "collate_fn"):
        return instantiate(dataloader_config.collate_fn)
    else:
        from espnet2.train.collate_fn import CommonCollateFn

        return CommonCollateFn(int_pad_value=-1)


def collect_stats_local(
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    output_dir: Path,
    task: Optional[str],
    write_collected_feats: bool,
    batch_size: int,
):
    """Collect statistics in single-process mode."""
    print(f"[{mode}] Running in local (non-parallel) mode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_espnet_model(task, model_config) if task else instantiate(model_config)
    model = model.to(device).eval()

    organizer = instantiate(dataset_config)
    dataset = getattr(organizer, mode)
    collate_fn = _build_collate_fn(dataloader_config)

    sum_dict, sq_dict, count_dict, writers = (
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        {},
    )

    index_batches = [
        list(range(i, min(i + batch_size, len(dataset))))
        for i in range(0, len(dataset), batch_size)
    ]

    with DatadirWriter(output_dir / mode) as datadir_writer:
        for idxs in tqdm(index_batches, desc=f"[{mode}]"):
            result = process_batch_batching(
                idxs,
                model=model,
                dataset=dataset,
                collate_fn=collate_fn,
                device=device,
                write_collected_feats=write_collected_feats,
            )
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
