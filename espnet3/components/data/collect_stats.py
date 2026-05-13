"""Collect statistics over a dataset using a model's feature extraction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.train.collate_fn import CommonCollateFn
from espnet3.parallel.base_runner import BaseRunner, cat_shard_files
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.task_utils import get_espnet_model

__all__ = [
    "CollectStatsInferenceProvider",
    "CollectStatsRunner",
    "collect_stats",
    "collect_stats_batch",
]


def collect_stats_batch(
    idxs: List[int],
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


def _build_collate_fn(dataloader_config):
    if not isinstance(dataloader_config, DictConfig):
        dataloader_config = (
            OmegaConf.create(dataloader_config)
            if dataloader_config is not None
            else OmegaConf.create({})
        )

    if (
        hasattr(dataloader_config, "collate_fn")
        and dataloader_config.collate_fn is not None
    ):
        return instantiate(dataloader_config.collate_fn)
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
    if not isinstance(dataset_config, DictConfig):
        dataset_config = OmegaConf.create(dataset_config)

    organizer = instantiate(dataset_config)
    dataset = getattr(organizer, mode, None)
    if dataset is None:
        raise ValueError(f"Dataset organizer does not provide split '{mode}'")
    return dataset


def _get_dataset_length(
    dataset_config, mode: str, shard_idx: Optional[int] = None
) -> int:
    dataset = _instantiate_dataset(dataset_config, mode)
    if shard_idx is not None:
        if not hasattr(dataset, "shard"):
            raise RuntimeError("Dataset does not support sharding")
        dataset = dataset.shard(shard_idx)
    return len(dataset)


def _persist_feats_for_key(
    writer: NpyScpWriter,
    feats: Dict[str, np.ndarray],
    feat_key: str,
    uids_in_order: List[str],
) -> None:
    feat_batch = feats[feat_key]
    len_batch = feats.get(f"{feat_key}_lengths", None)
    for b_idx, uid in enumerate(uids_in_order):
        seq = feat_batch[b_idx]
        if len_batch is not None:
            length = int(len_batch[b_idx])
            seq = seq[:length]
        else:
            seq = seq[None]
        if not isinstance(seq, np.ndarray):
            seq = np.asarray(seq)
        writer[uid] = seq


class CollectStatsInferenceProvider(EnvironmentProvider):
    """EnvironmentProvider tailored for collect-stats jobs.

    Beyond the usual dataset/model/collate_fn/device, the env exposes the
    ``write_collected_feats`` flag the reducer hooks on
    :class:`CollectStatsRunner` need. ``output_dir`` and ``shard_subdir`` are
    injected by the runner itself, so this provider stays focused on the
    inference environment.
    """

    def __init__(
        self,
        model_config,
        dataset_config,
        dataloader_config,
        mode: str,
        task: Optional[str] = None,
        shard_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = dict(),
    ):
        """Initialize CollectStatsInferenceProvider object."""
        config = OmegaConf.create({})
        config.model_config = model_config
        config.dataset_config = dataset_config
        config.dataloader_config = dataloader_config
        config.mode = mode
        config.task = task
        config.shard_idx = shard_idx
        config.update(**params)
        super().__init__(config)

    def build_env_local(self) -> Dict[str, Any]:
        """Build the environment once on the driver for local inference."""
        env: Dict[str, Any] = dict()
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
        env["write_collected_feats"] = bool(self.config.write_collected_feats)
        return env

    def build_worker_setup_fn(self):
        """Return a Dask worker setup function that builds dataset/model."""
        dataloader_config = self.config.dataloader_config
        config = self.config
        write_collected_feats = bool(self.config.write_collected_feats)

        def setup():
            env: Dict[str, Any] = dict()
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
            env["write_collected_feats"] = write_collected_feats
            return env

        return setup


class CollectStatsRunner(BaseRunner):
    """Runner that executes collect-stats over batches of indices.

    Implements only the four persistence hooks of :class:`BaseRunner`. Per-shard
    accumulators and writers live on the worker; the driver only sees a small
    summary per shard and a final scalar reduction + scp/shape concatenation
    runs in :py:meth:`merge_scalar` / :py:meth:`merge_shard_files`.
    """

    def __init__(
        self,
        provider: EnvironmentProvider,
        output_dir: str | Path,
        mode: str,
        write_collected_feats: bool = False,
        **kwargs,
    ):
        """Initialize CollectStatsRunner object."""
        super().__init__(
            provider,
            output_dir=output_dir,
            shard_subdir=mode,
            **kwargs,
        )
        self.mode = mode
        self.write_collected_feats = bool(write_collected_feats)

    @staticmethod
    def forward(
        batch_indices: Iterable[int] | int,
        dataset,
        model,
        collate_fn,
        device,
        write_collected_feats: bool = False,
        collect_stats_kwargs: Optional[Dict[str, Any]] = None,
        **env,
    ):
        """Process a batch of dataset indices and compute feature statistics."""
        if isinstance(batch_indices, Iterable) and not isinstance(
            batch_indices, (str, bytes)
        ):
            indices = [int(i) for i in batch_indices]
        else:
            indices = [int(batch_indices)]

        return collect_stats_batch(
            indices,
            model=model,
            dataset=dataset,
            collate_fn=collate_fn,
            device=device,
            write_collected_feats=write_collected_feats,
            collect_stats_kwargs=collect_stats_kwargs,
        )

    @staticmethod
    def open_writers(
        shard_dir: Optional[Path],
        write_collected_feats: bool = False,
        **env,
    ) -> Dict[str, Any]:
        """Open per-shard shape file handles and (optionally) NpyScpWriters.

        ``shard_dir`` is the unique ``split.<rank>/`` directory; required here
        because every batch streams shape lines and feature payloads directly
        to disk.
        """
        if shard_dir is None:
            raise RuntimeError(
                "CollectStatsRunner requires output_dir for shard outputs."
            )
        return {
            "_shard_dir": shard_dir,
            "_write_feats": bool(write_collected_feats),
            "shape_handles": {},
            "feat_writers": {},
        }

    @staticmethod
    def write_record(
        writers: Dict[str, Any],
        result,
        state: Dict[str, Any],
        **env,
    ) -> None:
        """Fold a batch's contribution into the shard state and stream to disk."""
        write_feats = writers["_write_feats"]
        shard_dir: Path = writers["_shard_dir"]

        if write_feats:
            stats, shape_info, feats = result
        else:
            stats, shape_info = result
            feats = None

        sum_acc = state.setdefault("sum", defaultdict(lambda: 0))
        sq_acc = state.setdefault("sq", defaultdict(lambda: 0))
        count_acc = state.setdefault("count", defaultdict(lambda: 0))
        for feat_key, agg in stats.items():
            sum_acc[feat_key] += agg["sum"]
            sq_acc[feat_key] += agg["sq"]
            count_acc[feat_key] += agg["count"]

        for feat_key, uid2shape in shape_info.items():
            handle = writers["shape_handles"].get(feat_key)
            if handle is None:
                handle = (shard_dir / f"{feat_key}_shape").open("w", encoding="utf-8")
                writers["shape_handles"][feat_key] = handle
            for uid, shape_str in uid2shape.items():
                handle.write(f"{uid} {shape_str}\n")

            if write_feats and feats is not None and feat_key in feats:
                writer = writers["feat_writers"].get(feat_key)
                if writer is None:
                    feat_root = shard_dir / "collect_feats"
                    writer = NpyScpWriter(
                        feat_root / f"data_{feat_key}",
                        feat_root / f"{feat_key}.scp",
                    )
                    writers["feat_writers"][feat_key] = writer
                _persist_feats_for_key(writer, feats, feat_key, list(uid2shape.keys()))

    @staticmethod
    def close_writers(writers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Close handles/writers and report which keys produced output files."""
        shape_handles = writers.get("shape_handles", {})
        feat_writers = writers.get("feat_writers", {})
        for handle in shape_handles.values():
            handle.close()
        for writer in feat_writers.values():
            writer.close()
        return {
            "shape_keys": list(shape_handles.keys()),
            "feat_keys_written": list(feat_writers.keys()),
        }

    def merge_scalar(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reduce sum/sq/count per feature key across shards."""
        sum_dict: Dict[str, Any] = defaultdict(lambda: 0)
        sq_dict: Dict[str, Any] = defaultdict(lambda: 0)
        count_dict: Dict[str, int] = defaultdict(lambda: 0)
        for state in states:
            for key, value in state.get("sum", {}).items():
                sum_dict[key] += value
            for key, value in state.get("sq", {}).items():
                sq_dict[key] += value
            for key, value in state.get("count", {}).items():
                count_dict[key] += value
        return {
            "sum": dict(sum_dict),
            "sq": dict(sq_dict),
            "count": dict(count_dict),
        }

    def merge_shard_files(
        self,
        shard_dirs: List[Path],
        states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Concatenate per-shard ``*_shape`` and ``collect_feats/*.scp`` files."""
        if self.output_dir is None:
            return {}
        shape_keys: set = set()
        feat_keys_written: set = set()
        for state in states:
            shape_keys.update(state.get("shape_keys", []))
            feat_keys_written.update(state.get("feat_keys_written", []))

        mode_dir = self.output_dir / self.mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for feat_key in shape_keys:
            cat_shard_files(
                shard_dirs, f"{feat_key}_shape", mode_dir / f"{feat_key}_shape"
            )

        if self.write_collected_feats and feat_keys_written:
            feat_dir = mode_dir / "collect_feats"
            feat_dir.mkdir(parents=True, exist_ok=True)
            for feat_key in feat_keys_written:
                cat_shard_files(
                    shard_dirs,
                    f"collect_feats/{feat_key}.scp",
                    feat_dir / f"{feat_key}.scp",
                )
        return {}


def _collect_stats_common(
    model_config,
    dataset_config,
    dataloader_config,
    mode: str,
    output_dir: Path,
    task: Optional[str],
    write_collected_feats: bool,
    batch_size: int,
    shard_idx: Optional[int] = None,
):
    num_items = _get_dataset_length(dataset_config, mode, shard_idx)
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
    runner = CollectStatsRunner(
        provider,
        output_dir=output_dir,
        mode=mode,
        write_collected_feats=write_collected_feats,
    )

    if not index_batches:
        return {}, {}, {}

    aggregated = runner(index_batches)
    return aggregated["sum"], aggregated["sq"], aggregated["count"]


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

    Runs the runner-based collection once, optionally configuring parallel
    execution via :func:`espnet3.parallel.set_parallel` when ``parallel_config``
    is provided.

    Args:
        model_config: Configuration object used to instantiate the model that
            extracts features from the input examples.
        dataset_config: Configuration of the dataset organizer providing the
            split specified by ``mode``.
        dataloader_config: Dataloader configuration.
        mode: Name of the dataset split to process (``train`` or ``valid``).
        output_dir: Directory where aggregated statistics and optionally
            collected features are written.
        task: Name of the ESPnet task. If ``None``, ``model_config`` should be
            directly instantiable.
        parallel_config: Configuration for parallel execution.
        write_collected_feats: Whether to persist the raw collected features.
        batch_size: Number of dataset items processed per batch.

    Returns:
        None: Aggregated statistics are saved under ``output_dir / mode``.
    """
    mode_config = getattr(dataloader_config, mode, None)
    if mode_config is not None and hasattr(mode_config, "multiple_iterator"):
        raise RuntimeError(
            "ESPnet3 does not support multiple_iterator. "
            "If you need sharding, select a shard explicitly "
            "(e.g., point the dataset/shape files to split.*) "
            "and run collect_stats on that shard."
        )
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

    mode_dir = Path(output_dir) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    for key in sum_dict:
        np.savez(
            mode_dir / f"{key}_stats.npz",
            count=count_dict[key],
            sum=sum_dict[key],
            sum_square=sq_dict[key],
        )
    with open(mode_dir / "stats_keys", "w") as f:
        f.write("\n".join(sum_dict) + "\n")
