# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Batching utilities for efficient data loading in SpeechLM training."""

import logging
import multiprocessing as mp
import random
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
_FINISH_RATIO = 0.995  # Batch is "complete" when >= 99% full
_N_STRATA = 64  # Number of length strata for diverse packing
_SEED = 42  # Random seed for shuffling in diverse packing


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


def _diverse_bfd_worker(items: List[tuple], batch_token: int) -> List[List[tuple]]:
    """Stratified Best Fit: diversity + efficiency via interleaved packing.

    Items are divided into length strata, shuffled within each stratum,
    and interleaved (long, short, long, short...) before Best Fit packing.
    This ensures each batch contains items from diverse length ranges while
    maintaining high packing efficiency.
    """

    if not items:
        return []

    rng = random.Random(_SEED)
    n = len(items)
    n_strata = min(_N_STRATA, n)

    # Sort and divide into length strata
    sorted_items = sorted(items, key=lambda x: x[0])
    strata = [[] for _ in range(n_strata)]
    for i, item in enumerate(sorted_items):
        strata[i * n_strata // n].append(item)

    # Shuffle within each stratum for epoch-to-epoch variety
    for s in strata:
        rng.shuffle(s)

    # Interleave: longest stratum first, alternate long/short
    # Order: [longest, shortest, 2nd longest, 2nd shortest, ...]
    reordered = []
    left, right = 0, n_strata - 1
    while left <= right:
        reordered.append(strata[right])  # long first for efficiency
        if left != right:
            reordered.append(strata[left])
        left += 1
        right -= 1

    # Round-robin across reordered strata
    interleaved = []
    max_len = max(len(s) for s in reordered)
    for i in range(max_len):
        for s in reordered:
            if i < len(s):
                interleaved.append(s[i])

    # Best Fit packing
    min_remaining = int((1.0 - _FINISH_RATIO) * batch_token)
    batches, active = [], SortedList()

    for length, key in interleaved:
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
    """Create batches using diverse Best Fit (parallel with 8 workers).

    Uses stratified interleaving to ensure length diversity within batches
    while maintaining high packing efficiency.

    Args:
        keys: List of sample keys to batch.
        key_to_length: Dictionary mapping each key to its length.
        batch_token: Maximum number of tokens allowed per batch.

    Returns:
        List of batches, where each batch is a list of keys.
    """
    # Convert to (length, key) tuples - avoids copying dict to workers
    items = [(key_to_length[k], k) for k in keys]

    # Skip multiprocessing for small inputs
    if len(keys) < _NUM_WORKERS:
        batches = _diverse_bfd_worker(items, batch_token)
        return [[key for _, key in batch] for batch in batches]

    # Split items → parallel (sort + pack) → merge
    chunks = [items[i::_NUM_WORKERS] for i in range(_NUM_WORKERS)]

    with mp.Pool(_NUM_WORKERS) as pool:
        results = pool.starmap(_diverse_bfd_worker, [(c, batch_token) for c in chunks])

    # Merge: keep complete batches, re-pack incomplete ones
    min_filled = int(_FINISH_RATIO * batch_token)
    complete, redo = [], []
    for batches in results:
        for b in batches:
            total = sum(length for length, _ in b)
            (complete if total >= min_filled else redo).append(b)

    if redo:
        redo_items = [item for b in redo for item in b]
        complete.extend(_diverse_bfd_worker(redo_items, batch_token))

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
    else:
        logger.info(f"No need to synchronize sharded dataset. #Batches: {n_batches}")

    return batches
