from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet3.parallel import get_client, parallel_for, set_parallel
from espnet3.task import get_espnet_model


# -----------------------------------------------------------------------------
# Worker setup (used for parallel_for)
# -----------------------------------------------------------------------------
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
    Create a setup_fn for Dask workers. The returned dict keys MUST match the
    parameter names of `process_batch_batching` (so they are auto-injected).

    Returns:
        Callable[[], dict]
    """

    def setup_fn():
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        model = (
            get_espnet_model(task, model_config) if task else instantiate(model_config)
        )
        model = model.to(device).eval()

        # Dataset (optionally sharded)
        organizer = instantiate(dataset_config)
        ds = getattr(organizer, mode)
        if shard_idx is not None:
            ds = ds.shard(shard_idx)

        # Collate function
        if hasattr(dataloader_config, "collate_fn"):
            collate_fn = instantiate(dataloader_config.collate_fn)
        else:
            from espnet2.train.collate_fn import CommonCollateFn

            collate_fn = CommonCollateFn(int_pad_value=-1)

        # Keys must match process_batch_batching arguments
        return {
            "model": model,
            "dataset": ds,
            "collate_fn": collate_fn,
            "device": device,
            "write_collected_feats": write_collected_feats,
        }

    return setup_fn


# -----------------------------------------------------------------------------
# Per-batch work (shared by local and parallel paths)
# -----------------------------------------------------------------------------
def process_batch_batching(
    idxs: list[int],
    *,
    model=None,
    dataset=None,
    collate_fn=None,
    device=None,
    write_collected_feats: bool = False,
):
    """
    Process a batch of dataset indices to compute feature statistics.

    This function runs on a Dask worker (parallel) or the driver (local).
    All worker-local state is injected via wrap_func_with_worker_env using
    keys provided by `setup_fn`.

    Args:
        idxs (List[int]): List of dataset indices to process.
        model: Model instance exposing `collect_feats(**batch)`.
        dataset: Dataset object where `dataset[i] -> (uid, sample_dict)`.
        collate_fn: Function that collates a list of (uid, sample) into a batch.
        device: torch.device where tensors should be placed.
        write_collected_feats (bool): If True, also return raw features.

    Returns:
        If write_collected_feats is True:
            (stats, shape_info, feats)
        Else:
            (stats, shape_info)

        - stats: Dict[feat_key] -> {"sum": np.ndarray, "sq": np.ndarray, "count": int}
        - shape_info: Dict[feat_key] -> Dict[uid] -> "d0,d1,..."
        - feats: Dict[feat_key] -> np.ndarray or List[np.ndarray] for this batch
    """
    # Prepare batch
    items = [dataset[i] for i in idxs]  # list of (uid, sample_dict)
    uids, samples = zip(*items)
    batch = collate_fn(items)
    # batch = (uids, tensors_dict) or similar; use tensors_dict
    tensors = {
        k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch[1].items()
    }

    # Run feature extraction
    with torch.no_grad():
        feats = model.collect_feats(**tensors)

    # Move to CPU numpy (keep non-tensor entries as-is)
    feats = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
        for k, v in feats.items()
    }

    stats = defaultdict(lambda: {"sum": 0, "sq": 0, "count": 0})
    shape_info = defaultdict(dict)

    # Aggregate per-sample stats and collect shapes
    uid_list = list(uids)
    for b_idx, uid in enumerate(uid_list):
        for feat_key in list(feats.keys()):
            # Handle *_lengths alignment if present
            if f"{feat_key}_lengths" in feats:
                length = int(feats[f"{feat_key}_lengths"][b_idx])
                seq = feats[feat_key][b_idx][:length]
            else:
                # Treat as single-frame feature (parity with old local path)
                seq = feats[feat_key][b_idx][None]

            stats[feat_key]["sum"] += seq.sum(0)
            stats[feat_key]["sq"] += (seq**2).sum(0)
            stats[feat_key]["count"] += len(seq)
            shape_info[feat_key][uid] = ",".join(map(str, seq.shape))

    if write_collected_feats:
        return stats, shape_info, feats
    else:
        return stats, shape_info


# -----------------------------------------------------------------------------
# Common accumulator/persister (shared by local / parallel / multi-iterator)
# -----------------------------------------------------------------------------
def _accumulate_and_persist_batch(
    *,
    stats: dict,
    shape_info: dict,
    feats: Optional[dict],
    sum_dict: dict,
    sq_dict: dict,
    count_dict: dict,
    datadir_writer: DatadirWriter,
    writers: dict,
    mode: str,
    output_dir: Path,
    write_collected_feats: bool,
    shape_key_suffix: str = "",
):
    """
    Merge per-batch stats into global aggregates and persist shapes/features.

    - Aggregates: sum_dict / sq_dict / count_dict
    - Writes shape info via DatadirWriter
    - Optionally writes features to NpyScpWriter (including *_lengths)
      using the same trimming/wrapping logic as local mode.
    - `shape_key_suffix` allows shard-specific shape keys (e.g., ".shard.0").
    """
    # 1) Aggregate stats
    for feat_key, agg in stats.items():
        sum_dict[feat_key] += agg["sum"]
        sq_dict[feat_key] += agg["sq"]
        count_dict[feat_key] += agg["count"]

    # 2) Persist shapes and (optionally) features
    for feat_key, uid2shape in shape_info.items():
        # Shapes (optionally suffix for shards)
        shape_key = f"{feat_key}_shape{shape_key_suffix}"
        for uid, shape_str in uid2shape.items():
            datadir_writer[shape_key][uid] = shape_str

        # Features: keep identical behavior to local mode
        if write_collected_feats and feats is not None and feat_key in feats:
            uids_in_order = list(uid2shape.keys())  # insertion order == batch order
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
                    # Trim by length when a paired *_lengths exists
                    L = int(len_batch[b_idx])
                    seq = seq[:L]
                else:
                    # Local parity: wrap scalar/1D as [None]
                    seq = seq[None]

                if not isinstance(seq, np.ndarray):
                    seq = np.asarray(seq)
                w[uid] = seq


# -----------------------------------------------------------------------------
# Local (single-process) collection reusing the same per-batch logic
# -----------------------------------------------------------------------------
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
    """
    Collect statistics in single-process mode using the same per-batch logic
    as the parallel path (process_batch_batching + _accumulate_and_persist_batch).
    """
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

    sum_dict, sq_dict, count_dict, writers = (
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        {},
    )

    # Build batches of indices, then reuse the same per-batch worker function
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


# -----------------------------------------------------------------------------
# Parallel (Dask) collection using setup_fn + parallel_for
# -----------------------------------------------------------------------------
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
    """
    Collect feature statistics using Dask parallel execution with `parallel_for`
    and worker `setup_fn` (no WorkerPlugin).
    """
    set_parallel(parallel_config)

    # Determine dataset length on driver only (do not keep instances)
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

    # Build setup_fn for workers
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
        # Use parallel_for with setup_fn; results stream in completion order
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


# -----------------------------------------------------------------------------
# Multiple-iterator (sharded) collection using setup_fn + parallel_for
# -----------------------------------------------------------------------------
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
    """
    Collect statistics on sharded datasets using Dask + parallel_for + setup_fn.

    Note:
        Current spec does NOT save raw features in multi-iterator mode
        (write_collected_feats=False). Only aggregates and shapes are written.
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
        # Determine shard length on driver
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
            write_collected_feats=False,  # per current spec
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
                # In multi-iterator we don't save feats; only stats + shapes
                shard_stats, shape_info = shard_result

                _accumulate_and_persist_batch(
                    stats=shard_stats,
                    shape_info=shape_info,
                    feats=None,
                    sum_dict=sum_dict,
                    sq_dict=sq_dict,
                    count_dict=count_dict,
                    datadir_writer=datadir_writer,
                    writers={},  # no feature writers in multi-iterator mode
                    mode=mode,
                    output_dir=output_dir,
                    write_collected_feats=False,
                    shape_key_suffix=f".shard.{shard_idx}",
                )

    return sum_dict, sq_dict, count_dict


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
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
    Entry point for collecting statistics of features from a dataset.

    Supports both local and distributed (Dask) execution, and optionally saving
    the features. Handles different modes including multiple iterator
    (sharded datasets).

    Args:
        model_config (DictConfig): Configuration for the model.
        dataset_config (DictConfig): Configuration for dataset loading.
        dataloader_config (DictConfig): Configuration for data loader
            (including collate_fn).
        mode (str): One of ['train', 'valid'].
        output_dir (Path): Output directory where stats and features will be stored.
        task (Optional[str]): ESPnet model task name, if used.
        parallel_config (Optional[dict]): If provided, enables parallel processing
            via Dask.
        write_collected_feats (bool): Whether to save the features to disk.
        batch_size (int): Batch size used for processing.

    Returns:
        None. Results are saved to disk under `output_dir / mode`.
    """
    # First check if we use multiple_iterator
    mode_config = getattr(dataloader_config, mode)
    if getattr(mode_config, "multiple_iterator", False):
        if parallel_config is None:
            raise RuntimeError("You should set parallel config with multiple iterator.")
        if write_collected_feats:
            raise ValueError(
                "Currently this option is not supported."
                " If you really want to save all feats at this stage,"
                " comment out this warning and add feats as return value."
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
        # ---- Local mode ----
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

    # Save aggregated statistics
    for key in sum_dict:
        np.savez(
            output_dir / mode / f"{key}_stats.npz",
            count=count_dict[key],
            sum=sum_dict[key],
            sum_square=sq_dict[key],
        )
    with open(output_dir / mode / "stats_keys", "w") as f:
        f.write("\n".join(sum_dict) + "\n")
