# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Batching utilities for efficient data loading in SpeechLM training."""

import logging
import multiprocessing as mp
from typing import Dict, List, TypeVar

import torch
import torch.distributed as dist
from sortedcontainers import SortedList

logger = logging.getLogger(__name__)

T = TypeVar("T")


def batchfy_bucket(
    keys: List[T], key_to_length: Dict[T, int], batch_token: int
) -> List[List[T]]:
    """Create batches using bucket batching strategy.

    Samples are sorted by length and grouped into buckets such that
    the total tokens (max_length * batch_size) does not exceed the
    batch_token limit.

    Args:
        keys: List of sample keys to batch.
        key_to_length: Dictionary mapping each key to its length.
        batch_token: Maximum number of tokens allowed per batch.

    Returns:
        List of buckets, where each bucket is a list of keys.
    """
    # Sort keys by length
    sorted_keys = sorted(keys, key=lambda k: key_to_length[k])

    buckets = []
    current_bucket = []

    for key in sorted_keys:
        key_length = key_to_length[key]

        if not current_bucket:
            # First item in bucket
            current_bucket.append(key)
        else:
            # Calculate total tokens if we add this key
            # Since sorted, key_length is the new max length
            new_size = len(current_bucket) + 1
            total_tokens = key_length * new_size

            if total_tokens <= batch_token:
                current_bucket.append(key)
            else:
                # Start a new bucket
                buckets.append(current_bucket)
                current_bucket = [key]

    # Add the last bucket if it's not empty
    if current_bucket:
        buckets.append(current_bucket)

    return buckets


_NUM_WORKERS = 8  # Fixed for reproducibility
_FINISH_RATIO = 0.99  # Batch is "complete" when >= 99% full


def _bfd_worker(items: List[tuple], batch_token: int) -> List[List[tuple]]:
    """Worker: sort (length, key) items and run Best Fit Decreasing."""
    sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
    batches, active = [], SortedList()
    min_remaining = int((1.0 - _FINISH_RATIO) * batch_token)

    for length, key in sorted_items:
        idx = active.bisect_left((length, -1))
        if idx < len(active):
            rem, bid = active.pop(idx)
            batches[bid].append((length, key))
            if rem - length > min_remaining:
                active.add((rem - length, bid))
        else:
            if batch_token - length > min_remaining:
                active.add((batch_token - length, len(batches)))
            batches.append([(length, key)])

    return batches


def batchfy_pack(
    keys: List[T], key_to_length: Dict[T, int], batch_token: int
) -> List[List[T]]:
    """Create batches using Best Fit Decreasing (parallel with 8 workers).

    Args:
        keys: List of sample keys to batch.
        key_to_length: Dictionary mapping each key to its length.
        batch_token: Maximum number of tokens allowed per batch.

    Returns:
        List of batches, where each batch is a list of keys.
    """

    # NOTE(Jinchuan): we observe some blocky loss fluctuation during training,
    # why may suggest the batchfy_pack would potentially have some issue. We
    # should revisit this issue a bit later.

    # Convert to (length, key) tuples - avoids copying dict to workers
    items = [(key_to_length[k], k) for k in keys]

    # Skip multiprocessing for small inputs
    if len(keys) < _NUM_WORKERS:
        batches = _bfd_worker(items, batch_token)
        return [[key for _, key in batch] for batch in batches]

    # Split items → parallel (sort + pack) → merge
    chunks = [items[i::_NUM_WORKERS] for i in range(_NUM_WORKERS)]

    with mp.Pool(_NUM_WORKERS) as pool:
        results = pool.starmap(_bfd_worker, [(c, batch_token) for c in chunks])

    # Merge: keep complete batches, re-pack incomplete ones
    min_filled = int(_FINISH_RATIO * batch_token)
    complete, redo = [], []
    for batches in results:
        for b in batches:
            total = sum(length for length, _ in b)
            (complete if total >= min_filled else redo).append(b)

    if redo:
        redo_items = [item for b in redo for item in b]
        complete.extend(_bfd_worker(redo_items, batch_token))

    # Extract keys only (discard lengths)
    return [[key for _, key in batch] for batch in complete]


def batchfy(
    keys: List[T],
    key_to_length: Dict[T, int],
    batch_token: int,
    batch_method: str,
) -> List[List[T]]:
    """Create batches using the specified batching method.

    Args:
        keys: List of sample keys to batch.
        key_to_length: Dictionary mapping each key to its length.
        batch_token: Maximum number of tokens allowed per batch.
        batch_method: Batching method to use ("bucket" or "pack").

    Returns:
        List of batches, where each batch is a list of keys.

    Raises:
        ValueError: If batch_method is invalid.

    Notes:
        Samples with length exceeding batch_token are automatically
        discarded and a warning is logged.
    """
    # Filter out samples that exceed batch_token
    valid_keys = []
    discarded_count = 0

    for key in keys:
        key_length = key_to_length[key]
        if key_length > batch_token:
            discarded_count += 1
        else:
            valid_keys.append(key)

    # Report discarded samples if any
    if discarded_count > 0:
        logger.warning(
            f"Discarded {discarded_count} samples (out of {len(keys)}) "
            f"that exceed batch_token limit ({batch_token})"
        )

    if batch_method == "bucket":
        batches = batchfy_bucket(valid_keys, key_to_length, batch_token)
    elif batch_method == "pack":
        batches = batchfy_pack(valid_keys, key_to_length, batch_token)
    else:
        raise ValueError(
            f"Invalid batch_method: {batch_method}. " f"Must be 'bucket' or 'pack'."
        )

    batches = synchronize_batches(batches)
    return batches


def synchronize_batches(batches: List[List[T]]) -> List[List[T]]:
    """Synchronize batches across all GPU ranks in distributed training.

    Ensures all GPU ranks have the same number of batches by duplicating
    the last few batches on ranks with fewer batches. This is useful for
    distributed training where each rank may have different numbers of
    batches due to data sharding.

    Args:
        batches: List of batches to synchronize.

    Returns:
        Synchronized list of batches with duplicates added if necessary.

    Notes:
        - If torch.distributed is not initialized, returns unchanged
        - If CUDA is not available, returns batches unchanged
        - Duplicates are taken from the end of the batch list
    """
    if not torch.cuda.is_available() or not dist.is_initialized():
        return batches

    n_batches = len(batches)
    n_batches_tensor = torch.tensor([n_batches], dtype=torch.long, device="cuda")
    n_batches_list = [
        torch.tensor([0], dtype=torch.long, device="cuda")
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(n_batches_list, n_batches_tensor)
    tgt_n_batches = max(t.item() for t in n_batches_list)

    if tgt_n_batches > n_batches:
        batches = batches + batches[-(tgt_n_batches - n_batches) :]
        logger.info("Synchronize sharded dataset across all process")
        logger.info(f"#Batches: {n_batches} -> {tgt_n_batches}")

    return batches
