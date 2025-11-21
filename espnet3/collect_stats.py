"""Collect statistics over a dataset using a model's feature extraction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.train.collate_fn import CommonCollateFn
from espnet3.parallel.parallel import set_parallel
from espnet3.runner.base_runner import BaseRunner
from espnet3.runner.env_provider import EnvironmentProvider
from espnet3.task import get_espnet_model

__all__ = [
    "CollectStatsInferenceProvider",
    "CollectStatsRunner",
    "collect_stats",
    "collect_stats_multiple_iterator",
    "batch_collect_stats",
]


def batch_collect_stats(
    idxs: List[int],
    *,
    model=None,
    dataset=None,
    collate_fn=None,
    device: Optional[torch.device] = None,
    write_collected_feats: bool = False,
    collect_stats_kwargs: Optional[Dict[str, Any]] = None,
):
    """Process a batch of dataset indices and compute feature statistics."""
    structured_items: List[Tuple[str, Any]] = []
    for i in idxs:
        item = dataset[i]
        # We assume dataset should be DataOrganizer in espnet3.
        if (
            hasattr(dataset, "use_espnet_preprocessor")
            and dataset.use_espnet_preprocessor
        ):
            # Then it is a tuple with (uid, dict) and type(uid) is str.
            uid, sample = item
        else:
            uid, sample = str(i), item
        structured_items.append((uid, sample))

    batch = collate_fn(structured_items)
    if not isinstance(batch, Sequence) or len(batch) != 2:
        raise RuntimeError(
            "collect_stats expects the collate function to return (uids, batch_dict)."
        )

    uids, features = batch  # type: ignore[misc]
    if not isinstance(features, dict):
        raise RuntimeError(
            "collect_stats expects collate_fn to return a mapping for batch tensors."
        )

    tensors = {k: v.to(device) for k, v in features.items()}

    extra_kwargs = dict(collect_stats_kwargs or {})
    conflict = set(extra_kwargs).intersection(tensors)
    if conflict:
        raise ValueError(
            "collect_stats kwargs conflict with batch tensors: " + ", ".join(conflict)
        )

    with torch.no_grad():
        feats = model.collect_feats(**{**tensors, **extra_kwargs})

    feats = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
        for k, v in feats.items()
    }

    stats = defaultdict(lambda: {"sum": 0, "sq": 0, "count": 0})
    shape_info = defaultdict(dict)

    for b_idx, uid in enumerate(list(uids)):
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
    cfg = dataloader_config
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg) if cfg is not None else OmegaConf.create({})

    if hasattr(cfg, "collate_fn") and cfg.collate_fn is not None:
        return instantiate(cfg.collate_fn)
    else:
        return CommonCollateFn(int_pad_value=-1)


def _build_dataset(config: DictConfig):
    dataset = _instantiate_dataset(config.dataset_config, config.mode)
    shard_idx = config.get("shard_idx")
    if shard_idx is not None:
        if not hasattr(dataset, "shard"):
            raise RuntimeError("Dataset does not support sharding")
        dataset = dataset.shard(shard_idx)

    if hasattr(dataset, "use_espnet_collator"):
        dataset.use_espnet_collator = True
    return dataset


def _build_model(config: DictConfig):
    model_config = config.model_config
    if not isinstance(model_config, DictConfig):
        model_config = OmegaConf.create(model_config)
    task = config.get("task")
    if task:
        model = get_espnet_model(task, model_config)
    else:
        model = instantiate(model_config)

    collect_fn = getattr(model, "collect_feats", None)
    if collect_fn is None or not callable(collect_fn):
        raise AttributeError(
            "Model is missing required callable 'collect_feats' method."
        )
    return model


def _chunk_indices(num_items: int, batch_size: int) -> List[List[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    batches = [
        list(range(i, min(i + batch_size, num_items)))
        for i in range(0, num_items, batch_size)
    ]
    return [b for b in batches if b]


def _instantiate_dataset(dataset_config, mode: str):
    cfg = dataset_config
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    organizer = instantiate(cfg)
    dataset = getattr(organizer, mode, None)
    if dataset is None:
        raise ValueError(f"Dataset organizer does not provide split '{mode}'")
    return dataset


def _dataset_length(dataset_config, mode: str, shard_idx: Optional[int] = None) -> int:
    dataset = _instantiate_dataset(dataset_config, mode)
    if shard_idx is not None:
        if not hasattr(dataset, "shard"):
            raise RuntimeError("Dataset does not support sharding")
        dataset = dataset.shard(shard_idx)
    return len(dataset)


class CollectStatsInferenceProvider(EnvironmentProvider):
    """EnvironmentProvider tailored for collect-stats jobs."""

    def __init__(
        self,
        *,
        model_config,
        dataset_config,
        dataloader_config,
        mode: str,
        task: Optional[str] = None,
        shard_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = dict(),
    ):
        """Initialize CollectStatsInferenceProvider object."""
        cfg = OmegaConf.create({})
        cfg.model_config = model_config
        cfg.dataset_config = dataset_config
        cfg.dataloader_config = dataloader_config
        cfg.mode = mode
        cfg.task = task
        cfg.shard_idx = shard_idx
        cfg.update(**params)
        super().__init__(cfg)

    def build_env_local(self) -> Dict[str, Any]:
        """Build the environment once on the driver for local inference."""
        env = dict()
        collate_fn = _build_collate_fn(self.config.dataloader_config)
        env["collate_fn"] = collate_fn

        dataset = _build_dataset(self.config)
        if hasattr(dataset, "use_espnet_collator"):
            dataset.use_espnet_collator = isinstance(collate_fn, CommonCollateFn)

        env["dataset"] = dataset

        device = env.get("device")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env["device"] = device

        env["model"] = _build_model(self.config).to(device).eval()
        env["write_collected_feats"] = self.config.write_collected_feats
        return env

    def make_worker_setup_fn(self):
        """Return a Dask worker setup function that builds dataset/model."""
        dataloader_config = self.config.dataloader_config
        config = self.config

        def setup():
            env = dict()
            collate_fn = _build_collate_fn(dataloader_config)
            env["collate_fn"] = collate_fn

            dataset = _build_dataset(config)
            if hasattr(dataset, "use_espnet_collator"):
                dataset.use_espnet_collator = isinstance(collate_fn, CommonCollateFn)
            env["dataset"] = dataset

            device = env.get("device")
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            env["device"] = device

            env["model"] = _build_model(config).to(device).eval()
            env["write_collected_feats"] = self.config.write_collected_feats
            return env

        return setup


class CollectStatsRunner(BaseRunner):
    """Runner that executes collect-stats over batches of indices."""

    @staticmethod
    def forward(
        batch_indices: Iterable[int] | int,
        *,
        dataset,
        model,
        collate_fn,
        device,
        write_collected_feats: bool = False,
        collect_stats_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Process a batch of dataset indices and compute feature statistics."""
        if isinstance(batch_indices, Iterable) and not isinstance(
            batch_indices, (str, bytes)
        ):
            indices = [int(i) for i in batch_indices]
        else:
            indices = [int(batch_indices)]

        return batch_collect_stats(
            indices,
            model=model,
            dataset=dataset,
            collate_fn=collate_fn,
            device=device,
            write_collected_feats=write_collected_feats,
            collect_stats_kwargs=collect_stats_kwargs,
        )


def _collect_stats_common(
    *,
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    output_dir: Path,
    task: Optional[str],
    write_collected_feats: bool,
    batch_size: int,
    shard_idx: Optional[int] = None,
    shape_key_suffix: str = "",
    sum_dict: Optional[Dict] = None,
    sq_dict: Optional[Dict] = None,
    count_dict: Optional[Dict] = None,
    writers: Optional[Dict] = None,
):
    num_items = _dataset_length(dataset_config, mode, shard_idx)
    index_batches = _chunk_indices(num_items, batch_size) if num_items else []

    provider = CollectStatsInferenceProvider(
        model_config=model_config,
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        mode=mode,
        task=task,
        shard_idx=shard_idx,
        params={"write_collected_feats": write_collected_feats},
    )
    runner = CollectStatsRunner(provider)

    sum_dict = sum_dict or defaultdict(lambda: 0)
    sq_dict = sq_dict or defaultdict(lambda: 0)
    count_dict = count_dict or defaultdict(lambda: 0)
    writers = writers or {}

    results = runner(index_batches)

    with DatadirWriter(output_dir / mode) as datadir_writer:
        for result in results:
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
                shape_key_suffix=shape_key_suffix,
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
    parallel_config: Any,
    write_collected_feats: bool = False,
):
    """Collect stats on sharded datasets using Dask + setup_fn."""
    set_parallel(parallel_config)

    sum_dict, sq_dict, count_dict = (
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
        defaultdict(lambda: 0),
    )

    mode_cfg = getattr(dataloader_config, mode)
    num_shards = mode_cfg.num_shards

    for shard_idx in range(num_shards):
        sum_dict, sq_dict, count_dict = _collect_stats_common(
            model_config=model_config,
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
            mode=mode,
            output_dir=output_dir,
            task=task,
            write_collected_feats=write_collected_feats,
            batch_size=batch_size,
            shard_idx=shard_idx,
            shape_key_suffix=f".shard.{shard_idx}",
            sum_dict=sum_dict,
            sq_dict=sq_dict,
            count_dict=count_dict,
        )

    return sum_dict, sq_dict, count_dict


def collect_stats(
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    output_dir: Path,
    task: Optional[str] = None,
    parallel_config: Optional[DictConfig] = None,
    write_collected_feats: bool = False,
    batch_size: int = 4,
):
    """Entry point for collecting dataset statistics used for feature normalization.

    Depending on ``dataloader_config`` this function either:

    - Runs the runner-based collection once, optionally configuring parallel
      execution via :func:`espnet3.parallel.set_parallel` when ``parallel_config``
      is provided.
    - Handles multi-iterator (sharded) datasets by iterating over shards.

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
        sum_dict, sq_dict, count_dict = collect_stats_multiple_iterator(
            model_config,
            dataset_config,
            dataloader_config,
            mode,
            output_dir,
            task,
            batch_size,
            parallel_config,
            write_collected_feats,
        )

    else:
        if parallel_config is not None:
            set_parallel(parallel_config)
        sum_dict, sq_dict, count_dict = _collect_stats_common(
            model_config=model_config,
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
            mode=mode,
            output_dir=output_dir,
            task=task,
            write_collected_feats=write_collected_feats,
            batch_size=batch_size,
        )

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
