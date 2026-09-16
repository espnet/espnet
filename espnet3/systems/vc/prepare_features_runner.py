"""Runner for the parallel ``prepare_features`` stage of ``VCSystem``."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch

from espnet3.parallel.base_runner import BaseRunner, concatenate_shard_files
from espnet3.systems.vc.models.knnvc.matcher import match_features

logger = logging.getLogger(__name__)

FEATURE_SUFFIX = ".npy"
FEATURES_SCP_NAME = "feats.scp"


class PoolFeatureCache(threading.local):
    """Per-thread cache holding one prematching pool's encoded features.

    The provider puts a single instance in the worker environment, which
    ``BaseRunner`` shares across every ``forward`` call on that worker. A Dask
    worker runs several task threads by default, so two shards covering
    different pools can touch this object concurrently; subclassing
    ``threading.local`` gives each thread its own ``pool_key``/``feats`` pair.
    Without it, one thread's ``feats`` could be reset or read while another
    thread is filling it, which would silently mix two pools' frames into one
    matching set.

    Attributes:
        pool_key: Pool key whose features are currently cached, or ``None``.
        feats: Mapping of dataset index to that utterance's encoded features.
    """

    def __init__(self) -> None:
        """Initialize an empty cache for the calling thread."""
        super().__init__()
        self.pool_key: Optional[str] = None
        self.feats: Dict[int, torch.Tensor] = {}


class PrepareFeaturesRunner(BaseRunner):
    """Encode every utterance and, optionally, prematch it within its pool.

    This is the ESPnet3 port of kNN-VC's ``prematch_dataset.py``. For each
    utterance ``u`` with pool key ``p`` (``dataset.get_pool_key(idx)``, e.g. a
    speaker or a speaker/chapter id):

    1. encode ``u`` with the frozen encoder (``pad_to_hop=True`` so the frame
       count matches the waveform length for vocoder training);
    2. if ``prematch`` is enabled, build the pool of features of *all other*
       utterances with key ``p`` and replace each frame of ``u`` by the mean of
       its ``topk`` nearest pool frames (:func:`match_features`);
    3. write the result as float16 ``<features_dir>/<feature_name>.npy`` and
       one ``<feature_name> <path>`` line into the shard-local ``feats.scp``.

    The driver sorts indices by pool key before sharding, so consecutive items
    on a worker share a pool. Each worker thread keeps the current pool's
    encoded utterances in ``env["pool_cache"]`` (a :class:`PoolFeatureCache`)
    and drops them when the key changes, which bounds memory to one pool per
    thread and encodes each utterance once per pool.

    :meth:`merge` concatenates the shard ``feats.scp`` files into
    ``<features_dir>/feats.scp``. Re-running the stage is safe: finished
    shards are skipped by ``BaseRunner`` and feature files are overwritten
    atomically.
    """

    @staticmethod
    def _encode(idx: int, dataset, model) -> torch.Tensor:
        sample = dataset[idx]
        if "speech" not in sample:
            raise KeyError(
                "prepare_features dataset items must contain 'speech'; "
                f"got keys {sorted(sample.keys())}"
            )
        return model.encode(sample["speech"], pad_to_hop=True)

    @staticmethod
    def _encode_pool_features(
        pool_key: str,
        pool_indices: Dict[str, List[int]],
        dataset,
        model,
        pool_cache: "PoolFeatureCache",
    ) -> Dict[int, torch.Tensor]:
        """Return ``{idx: feats}`` for every utterance with ``pool_key``.

        Encoding happens once per utterance per pool: the calling thread's
        slice of ``pool_cache`` is reused while consecutive items share a pool
        key and dropped as soon as the key changes, which bounds memory to one
        pool.
        """
        if pool_cache.pool_key != pool_key:
            pool_cache.pool_key = pool_key
            pool_cache.feats = {}
        feats_by_idx: Dict[int, torch.Tensor] = pool_cache.feats
        for other in pool_indices[pool_key]:
            if other not in feats_by_idx:
                feats_by_idx[other] = (
                    PrepareFeaturesRunner._encode(other, dataset, model).half().cpu()
                )
        return feats_by_idx

    @staticmethod
    def _process_one(
        idx: int,
        dataset,
        model,
        pool_indices: Dict[str, List[int]],
        prematch: bool,
        topk: int,
        pool_cache: "PoolFeatureCache",
    ) -> Dict[str, Any]:
        pool_key = str(dataset.get_pool_key(idx))
        feature_name = str(dataset.get_feature_name(idx))

        if not prematch:
            feats = PrepareFeaturesRunner._encode(idx, dataset, model)
        else:
            feats_by_idx = PrepareFeaturesRunner._encode_pool_features(
                pool_key, pool_indices, dataset, model, pool_cache
            )
            source = feats_by_idx[idx].to(model.device, dtype=torch.float32)
            pool = [f for other, f in feats_by_idx.items() if other != idx]
            if not pool:
                logger.warning(
                    "Pool %s has a single utterance (%s); writing unmatched "
                    "features because there is nothing to prematch against.",
                    pool_key,
                    feature_name,
                )
                feats = source
            else:
                matching_pool = torch.cat(pool, dim=0).to(
                    model.device, dtype=torch.float32
                )
                feats = match_features(source, matching_pool, topk=topk)

        return {
            "idx": int(idx),
            "pool_key": pool_key,
            "feature_name": feature_name,
            "feats": feats.detach().cpu().half().numpy(),
        }

    @staticmethod
    def forward(
        idx: Union[int, Iterable[int]],
        dataset=None,
        model=None,
        pool_indices: Optional[Dict[str, List[int]]] = None,
        prematch: bool = True,
        topk: int = 4,
        pool_cache: Optional["PoolFeatureCache"] = None,
        **env,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Encode (and prematch) one index or a batch of indices.

        Args:
            idx: Dataset index or iterable of indices.
            dataset: Audio dataset implementing the ``prepare_features``
                contract (see ``PrepareFeaturesProvider``).
            model: Encoder with ``encode(speech, pad_to_hop) -> (T, D)`` and a
                ``device`` attribute.
            pool_indices: ``{pool_key: [idx, ...]}`` for pool construction.
            prematch: Whether to apply within-pool kNN prematching.
            topk: ``k`` for prematching.
            pool_cache: Per-thread cache of the current pool's features. A
                fresh one is created when omitted, which only costs re-encoding.
            **env: Ignored extra environment entries.

        Returns:
            ``{"idx", "pool_key", "feature_name", "feats"}`` (``feats`` is a
            float16 numpy array of shape ``(frames, dim)``), or a list of such
            dicts for a batch.
        """
        if pool_cache is None:
            pool_cache = PoolFeatureCache()
        if pool_indices is None:
            raise RuntimeError("pool_indices missing from prepare_features env.")
        if isinstance(idx, (int, np.integer)):
            return PrepareFeaturesRunner._process_one(
                int(idx), dataset, model, pool_indices, prematch, topk, pool_cache
            )
        return [
            PrepareFeaturesRunner._process_one(
                int(i), dataset, model, pool_indices, prematch, topk, pool_cache
            )
            for i in idx
        ]

    @staticmethod
    def open_writers(
        shard_dir: Optional[Path], features_dir: str = "", **env
    ) -> Dict[str, Any]:
        """Open the shard-local ``feats.scp`` writer."""
        if not features_dir:
            raise RuntimeError("features_dir missing from prepare_features env.")
        scp_path = Path(shard_dir) / FEATURES_SCP_NAME
        return {
            "scp": scp_path.open("w", encoding="utf-8"),
            "features_dir": Path(features_dir),
        }

    @staticmethod
    def write_record(
        writers: Dict[str, Any], result: Any, state: Dict[str, Any], **env
    ) -> None:
        """Write feature file(s) for one ``forward`` result and log them in SCP."""
        records = result if isinstance(result, list) else [result]
        features_dir: Path = writers["features_dir"]
        for record in records:
            output_path = features_dir / (record["feature_name"] + FEATURE_SUFFIX)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.npy")
            np.save(tmp_path, record["feats"])
            tmp_path.replace(output_path)
            writers["scp"].write(f"{record['feature_name']} {output_path}\n")

    @staticmethod
    def close_writers(
        writers: Dict[str, Any], state: Dict[str, Any], **env
    ) -> Optional[Dict[str, Any]]:
        """Close the shard-local SCP writer."""
        writers["scp"].close()
        return None

    def merge(self, shard_dirs: List[Path]) -> Path:
        """Concatenate shard ``feats.scp`` files into one SCP under ``features_dir``.

        The merged file is ``<features_dir>/feats.<shard_subdir>.scp`` when the
        runner was given a ``shard_subdir`` (the dataset entry name), else
        ``<features_dir>/feats.scp``.

        Args:
            shard_dirs: Completed shard directories.

        Returns:
            Path of the merged SCP file.
        """
        features_dir = Path(self.provider.params["features_dir"])
        ordered = sorted(shard_dirs, key=lambda p: int(p.name.split(".", 1)[1]))
        scp_name = (
            f"feats.{self.shard_subdir}.scp" if self.shard_subdir else FEATURES_SCP_NAME
        )
        out_path = features_dir / scp_name
        concatenate_shard_files(ordered, FEATURES_SCP_NAME, out_path)
        return out_path
