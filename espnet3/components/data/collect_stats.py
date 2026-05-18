"""Collect per-feature statistics for espnet3 datasets."""

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
from espnet3.parallel.base_runner import BaseRunner, concatenate_shard_files
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
    """Collect feature statistics for one batch of dataset indices.

    This function is the low-level batch worker used by
    :func:`collect_stats`. It reads dataset items, builds one collated batch,
    calls ``model.collect_feats(...)``, and accumulates per-feature sums,
    squared sums, counts, and shape metadata.

    Args:
        idxs: Dataset indices to process as one batch.
        model: Model instance that provides a callable ``collect_feats``
            method.
        dataset: Dataset or dataset-like object indexed by ``idxs``.
        collate_fn: Collate function that returns ``(uids, batch_dict)``.
        device: Device used for tensor inputs passed to ``model.collect_feats``.
        write_collected_feats: Whether to return the collected feature arrays in
            addition to aggregated statistics.
        collect_stats_kwargs: Extra keyword arguments forwarded to
            ``model.collect_feats``. Keys must not overlap with collated batch
            tensor names.

    Returns:
        tuple: ``(stats, shape_info)`` when ``write_collected_feats`` is
        ``False``. Returns ``(stats, shape_info, feats)`` when it is ``True``.
        ``stats`` stores per-feature ``sum``, ``sq``, and ``count`` values.
        ``shape_info`` maps each feature key to ``uid -> shape`` strings.

    Raises:
        RuntimeError: If ``collate_fn`` does not return ``(uids, batch_dict)``.
        ValueError: If ``collect_stats_kwargs`` conflicts with batch tensor
            names.

    Notes:
        If the dataset exposes ``use_espnet_preprocessor=True``, this function
        expects each dataset item to be ``(uid, sample)``.

    Examples:
        stats, shape_info = collect_stats_batch(
            [0, 1, 2, 3],
            model=model,
            dataset=dataset,
            collate_fn=collate_fn,
            device=torch.device("cpu"),
        )
    """
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

    for batch_idx, uid in enumerate(list(uids)):
        for feat_key in list(feats.keys()):
            if f"{feat_key}_lengths" in feats:
                length = int(feats[f"{feat_key}_lengths"][batch_idx])
                seq = feats[feat_key][batch_idx][:length]
            else:
                seq = feats[feat_key][batch_idx][None]

            stats[feat_key]["sum"] += seq.sum(0)
            stats[feat_key]["sq"] += (seq**2).sum(0)
            stats[feat_key]["count"] += len(seq)
            shape_info[feat_key][uid] = ",".join(map(str, seq.shape))

    if write_collected_feats:
        return stats, shape_info, feats
    else:
        return stats, shape_info


def _build_collate_fn(dataloader_config):
    """Instantiate the configured collate function or use the default.

    Args:
        dataloader_config: Dataloader config, either a ``DictConfig`` or a
            plain dict. If it has a non-``None`` ``collate_fn`` field, that
            function is instantiated via Hydra. Otherwise,
            ``CommonCollateFn(int_pad_value=-1)`` is returned.

    Returns:
        Callable: The collate function to use in batch assembly.
    """
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
    """Build the dataset split and apply shard selection when requested.

    Args:
        config: Provider config containing ``dataset_config``, ``mode``, and
            an optional ``shard_idx``. If ``shard_idx`` is set, the dataset
            must expose a ``shard(idx)`` method.

    Returns:
        Dataset instance for the requested split, optionally narrowed to one
        shard.

    Raises:
        RuntimeError: If ``shard_idx`` is set but the dataset has no
            ``shard`` method.
    """
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
    """Instantiate the model and validate ``collect_feats`` support.

    Args:
        config: Provider config containing ``model_config`` and an optional
            ``task`` string. When ``task`` is set, the model is resolved
            through the ESPnet task bridge via :func:`get_espnet_model`.

    Returns:
        The instantiated model with a callable ``collect_feats`` method.

    Raises:
        AttributeError: If the instantiated model does not expose a callable
            ``collect_feats`` method.
    """
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
    """Split ``range(num_items)`` into non-empty batches.

    Args:
        num_items: Total number of dataset items.
        batch_size: Maximum size of each batch. Must be a positive integer.

    Returns:
        List of index lists, each of length at most ``batch_size``. Empty
        batches are excluded.

    Raises:
        ValueError: If ``batch_size`` is not a positive integer.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    batches = [
        list(range(i, min(i + batch_size, num_items)))
        for i in range(0, num_items, batch_size)
    ]
    return [b for b in batches if b]


def _instantiate_dataset(dataset_config, mode: str):
    """Instantiate the dataset organizer and select one split.

    Args:
        dataset_config: Hydra-compatible config for the dataset organizer.
        mode: Name of the split attribute to retrieve (e.g. ``"train"``).

    Returns:
        The dataset split object exposed as ``organizer.<mode>``.

    Raises:
        ValueError: If the organizer does not expose an attribute named
            ``mode``.
    """
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
    """Return the number of items in one dataset split or shard.

    Args:
        dataset_config: Hydra-compatible config for the dataset organizer.
        mode: Name of the split to measure (e.g. ``"train"``).
        shard_idx: If set, the split is narrowed to this shard before
            measuring. Requires the dataset to expose a ``shard`` method.

    Returns:
        int: Number of items in the split or shard.

    Raises:
        RuntimeError: If ``shard_idx`` is given but the dataset has no
            ``shard`` method.
    """
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
    """Write one collected feature key to its SCP-backed numpy store.

    Args:
        writer: Open ``NpyScpWriter`` for the target feature key.
        feats: Batch of collected features keyed by feature name. An
            optional ``{feat_key}_lengths`` entry is used to trim padding.
        feat_key: The feature key to write from ``feats``.
        uids_in_order: Utterance IDs corresponding to the batch dimension,
            in the same order as the batch axis of ``feats[feat_key]``.
    """
    feat_batch = feats[feat_key]
    len_batch = feats.get(f"{feat_key}_lengths", None)
    for batch_idx, uid in enumerate(uids_in_order):
        seq = feat_batch[batch_idx]
        if len_batch is not None:
            length = int(len_batch[batch_idx])
            seq = seq[:length]
        else:
            seq = seq[None]
        if not isinstance(seq, np.ndarray):
            seq = np.asarray(seq)
        writer[uid] = seq


class CollectStatsInferenceProvider(EnvironmentProvider):
    """Build collect-stats execution environments.

    This provider prepares the dataset, collate function, device, and model for
    :class:`CollectStatsRunner`. It supports both local execution and worker
    setup for parallel jobs.

    Examples:
        provider = CollectStatsInferenceProvider(
            model_config=model_config,
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
            mode="train",
            task="asr",
        )
    """

    def __init__(
        self,
        model_config,
        dataset_config,
        dataloader_config,
        mode: str,
        task: Optional[str] = None,
        shard_idx: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the provider configuration.

        Args:
            model_config: Config used to instantiate the model.
            dataset_config: Config used to instantiate the dataset organizer.
            dataloader_config: Dataloader config, including optional
                ``collate_fn`` settings.
            mode: Dataset split name such as ``train`` or ``valid``.
            task: ESPnet task name. When set, the model is resolved through the
                espnet2 task bridge.
            shard_idx: Optional shard index applied to shardable datasets.
            params: Extra config values merged into the provider config. This is
                typically used for flags such as ``write_collected_feats``.
        """
        config = OmegaConf.create({})
        config.model_config = model_config
        config.dataset_config = dataset_config
        config.dataloader_config = dataloader_config
        config.mode = mode
        config.task = task
        config.shard_idx = shard_idx
        config.update(**(params or {}))
        super().__init__(config)

    def build_env_local(self) -> Dict[str, Any]:
        """Build the local execution environment once on the driver.

        Returns:
            dict: Environment mapping with keys ``collate_fn``, ``dataset``,
            ``device``, ``model``, and ``write_collected_feats``.
        """
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

    def build_worker_setup_fn(self):
        """Return a worker setup function for parallel collect-stats jobs.

        Returns:
            Callable: A zero-argument function that builds and returns the
            same environment dict as :meth:`build_env_local`. Called once
            per parallel worker process.
        """
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
    """Execute collect-stats batches and merge shard outputs.

    The runner delegates batch execution to :func:`collect_stats_batch`,
    persists per-shard metadata and optional collected features, and merges the
    shard outputs into the final mode directory.
    """

    def __init__(
        self,
        provider: EnvironmentProvider,
        output_dir: str | Path,
        mode: str,
        write_collected_feats: bool = False,
        **kwargs,
    ):
        """Initialize the runner.

        Args:
            provider: Environment provider that builds the dataset and model.
            output_dir: Root directory for collect-stats outputs.
            mode: Dataset split name used as the shard subdirectory.
            write_collected_feats: Whether to persist raw collected features.
            **kwargs: Extra arguments forwarded to :class:`BaseRunner`.
        """
        super().__init__(
            provider,
            output_dir=output_dir,
            shard_subdir=mode,
            **kwargs,
        )
        self.mode = mode
        self.write_collected_feats = write_collected_feats

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
        """Run collect-stats for one batch index group.

        Args:
            batch_indices: One or more dataset indices forming a single batch.
            dataset: Dataset indexed by the provided indices.
            model: Model with a callable ``collect_feats`` method.
            collate_fn: Collate function returning ``(uids, batch_dict)``.
            device: Device to move input tensors onto before inference.
            write_collected_feats: Whether to include raw feature arrays in
                the return value.
            collect_stats_kwargs: Extra keyword arguments forwarded to
                ``model.collect_feats``.
            **env: Additional environment keys (ignored).

        Returns:
            tuple: ``(stats, shape_info)`` or ``(stats, shape_info, feats)``
            as returned by :func:`collect_stats_batch`.
        """
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
        """Create per-shard writer state.

        Args:
            shard_dir: Directory where shard output files are written.
            write_collected_feats: Whether to open feature writers in
                addition to shape-file handles.
            **env: Additional environment keys (ignored).

        Returns:
            dict: Initial writers state with keys ``_shard_dir``,
            ``_write_feats``, ``shape_handles``, and ``feat_writers``.
        """
        return {
            "_shard_dir": shard_dir,
            "_write_feats": write_collected_feats,
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
        """Accumulate one batch result into shard files and in-memory state.

        Args:
            writers: Writer state returned by :meth:`open_writers`. Updated
                in-place with running sums, shape file handles, and optional
                feature writers.
            result: Return value of :meth:`forward` — either
                ``(stats, shape_info)`` or ``(stats, shape_info, feats)``.
            state: Persistent in-memory accumulator for ``sum``, ``sq``,
                and ``count`` across batches.
            **env: Additional environment keys (ignored).
        """
        writers["_state"] = state
        write_feats = writers["_write_feats"]
        shard_dir: Path = writers["_shard_dir"]

        if write_feats:
            stats, shape_info, feats = result
        else:
            stats, shape_info = result
            feats = None

        sum_acc = state.setdefault("sum", {})
        sq_acc = state.setdefault("sq", {})
        count_acc = state.setdefault("count", {})
        for feat_key, agg in stats.items():
            if feat_key in sum_acc:
                sum_acc[feat_key] += agg["sum"]
                sq_acc[feat_key] += agg["sq"]
                count_acc[feat_key] += agg["count"]
            else:
                sum_acc[feat_key] = agg["sum"]
                sq_acc[feat_key] = agg["sq"]
                count_acc[feat_key] = agg["count"]

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
        """Close shard writers and flush shard summary files.

        Writes ``shape_keys.txt``, ``feat_keys_written.txt``,
        ``stats_keys.txt``, and ``{key}_stats.npz`` for every accumulated
        feature key to the shard directory.

        Args:
            writers: Writer state produced by :meth:`open_writers` and
                populated by :meth:`write_record`.

        Returns:
            None
        """
        shard_dir = writers["_shard_dir"]
        shape_handles = writers.get("shape_handles", {})
        feat_writers = writers.get("feat_writers", {})
        for handle in shape_handles.values():
            handle.close()
        for writer in feat_writers.values():
            writer.close()
        shape_keys = sorted(shape_handles.keys())
        feat_keys_written = sorted(feat_writers.keys())
        state = writers.get("_state", {})
        stats_keys = sorted(state.get("sum", {}).keys())
        (shard_dir / "shape_keys.txt").write_text(
            "\n".join(shape_keys) + ("\n" if shape_keys else ""),
            encoding="utf-8",
        )
        (shard_dir / "feat_keys_written.txt").write_text(
            "\n".join(feat_keys_written) + ("\n" if feat_keys_written else ""),
            encoding="utf-8",
        )
        (shard_dir / "stats_keys.txt").write_text(
            "\n".join(stats_keys) + ("\n" if stats_keys else ""),
            encoding="utf-8",
        )
        for key in stats_keys:
            np.savez(
                shard_dir / f"{key}_stats.npz",
                count=state["count"][key],
                sum=state["sum"][key],
                sum_square=state["sq"][key],
            )
        return None

    def merge(self, shard_dirs: List[Path]) -> Dict[str, Any]:
        """Merge shard outputs into aggregated statistics for one split.

        Args:
            shard_dirs: List of shard directories produced by
                :meth:`close_writers`, one per parallel shard.

        Returns:
            dict: Aggregated totals with keys ``"sum"``, ``"sq"``, and
            ``"count"``, each mapping feature key to its accumulated value.
            Shape files and optional collected-feat SCP files are also
            concatenated into ``output_dir / mode``.
        """
        shape_keys: set = set()
        feat_keys_written: set = set()
        stats_keys: set = set()
        sum_dict: Dict[str, Any] = defaultdict(lambda: 0)
        sq_dict: Dict[str, Any] = defaultdict(lambda: 0)
        count_dict: Dict[str, int] = defaultdict(lambda: 0)
        for shard_dir in shard_dirs:
            shape_keys.update(
                (shard_dir / "shape_keys.txt").read_text(encoding="utf-8").splitlines()
                if (shard_dir / "shape_keys.txt").exists()
                else []
            )
            feat_keys_written.update(
                (shard_dir / "feat_keys_written.txt")
                .read_text(encoding="utf-8")
                .splitlines()
                if (shard_dir / "feat_keys_written.txt").exists()
                else []
            )
            for key in (
                (shard_dir / "stats_keys.txt").read_text(encoding="utf-8").splitlines()
                if (shard_dir / "stats_keys.txt").exists()
                else []
            ):
                stats_keys.add(key)
                data = np.load(shard_dir / f"{key}_stats.npz")
                sum_dict[key] += data["sum"]
                sq_dict[key] += data["sum_square"]
                count_dict[key] += data["count"]

        mode_dir = self.output_dir / self.mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for feat_key in shape_keys:
            concatenate_shard_files(
                shard_dirs, f"{feat_key}_shape", mode_dir / f"{feat_key}_shape"
            )

        if self.write_collected_feats and feat_keys_written:
            feat_dir = mode_dir / "collect_feats"
            feat_dir.mkdir(parents=True, exist_ok=True)
            for feat_key in feat_keys_written:
                concatenate_shard_files(
                    shard_dirs,
                    f"collect_feats/{feat_key}.scp",
                    feat_dir / f"{feat_key}.scp",
                )
        return {
            "sum": dict(sum_dict),
            "sq": dict(sq_dict),
            "count": dict(count_dict),
        }


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
    """Run collect-stats once and return aggregated in-memory totals.

    Args:
        model_config: Config used to instantiate the model.
        dataset_config: Config used to instantiate the dataset organizer.
        dataloader_config: Dataloader config forwarded to the provider.
        mode: Dataset split name (e.g. ``"train"``).
        output_dir: Root directory for shard and merged outputs.
        task: ESPnet task name, or ``None`` for direct instantiation.
        write_collected_feats: Whether to persist raw collected features.
        batch_size: Number of items per batch.
        shard_idx: Optional shard index for shardable datasets.

    Returns:
        tuple: ``(sum_dict, sq_dict, count_dict)`` — per-feature accumulated
        sums, squared sums, and item counts. All values are ``{}`` when the
        dataset is empty.
    """
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
    """Collect dataset statistics used by feature normalization stages.

    This is the public entry point for espnet3 collect-stats execution. It
    builds batches from the selected dataset split, runs
    ``model.collect_feats(...)`` over the full split, and writes aggregated
    ``*_stats.npz`` files under ``output_dir / mode``. When requested, it also
    writes SCP-backed collected feature dumps under ``collect_feats/``.

    Args:
        model_config: Configuration object used to instantiate the model that
            extracts features from the input examples.
        dataset_config: Configuration of the dataset organizer providing the
            split specified by ``mode``.
        dataloader_config: Dataloader configuration. If ``<mode>`` contains
            ``multiple_iterator``, this function raises because espnet3 does not
            support that mode here.
        mode: Name of the dataset split to process (``train`` or ``valid``).
        output_dir: Directory where aggregated statistics and optionally
            collected features are written.
        task: Name of the ESPnet task. If ``None``, ``model_config`` should be
            directly instantiable.
        parallel_config: Configuration for parallel execution.
        write_collected_feats: Whether to persist the raw collected features.
        batch_size: Number of dataset items processed per batch.

    Returns:
        None: This function writes outputs to disk and does not return the
        aggregated arrays.

    Raises:
        RuntimeError: If the selected dataloader mode uses
            ``multiple_iterator``.

    Notes:
        Output files are written under ``output_dir / mode``. For each feature
        key, the function writes ``{key}_stats.npz`` with ``count``, ``sum``,
        and ``sum_square`` arrays. It also writes a ``stats_keys`` file listing
        the aggregated feature keys.

    Examples:
        collect_stats(
            model_config=model_config,
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
            mode="train",
            output_dir=Path("exp/asr_stats"),
            task="asr",
            batch_size=8,
        )
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

    mode_dir = output_dir / mode
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
