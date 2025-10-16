"""Batching utilities for SpeechLM data loading."""

from typing import Dict, List, TypeVar

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


def batchfy_pack(
    keys: List[T], key_to_length: Dict[T, int], batch_token: int
) -> List[List[T]]:
    """Create batches using pack batching strategy.

    Uses Best Fit Decreasing algorithm to maximize batch utilization.
    Samples are sorted by length (descending) and packed into batches
    by finding the batch with minimum remaining space that can fit
    each sample. Batches at 99% capacity are marked as finished.

    Args:
        keys: List of sample keys to batch.
        key_to_length: Dictionary mapping each key to its length.
        batch_token: Maximum number of tokens allowed per batch.

    Returns:
        List of batches, where each batch is a list of keys.
    """
    # Sort keys by length in descending order (largest first)
    sorted_keys = sorted(keys, key=lambda k: key_to_length[k], reverse=True)

    finished_batches = []
    active_batches = []
    active_totals = []
    threshold = 0.99 * batch_token

    for key in sorted_keys:
        key_length = key_to_length[key]

        # Find the best active batch (minimum remaining space)
        best_batch_idx = -1
        min_remaining = float("inf")

        for idx, total in enumerate(active_totals):
            remaining = batch_token - total
            if remaining >= key_length and remaining < min_remaining:
                best_batch_idx = idx
                min_remaining = remaining

        if best_batch_idx >= 0:
            # Add to existing active batch
            active_batches[best_batch_idx].append(key)
            active_totals[best_batch_idx] += key_length

            # Check if batch is now finished (>= 99% full)
            if active_totals[best_batch_idx] >= threshold:
                finished_batches.append(active_batches[best_batch_idx])
                del active_batches[best_batch_idx]
                del active_totals[best_batch_idx]
        else:
            # Create new active batch
            active_batches.append([key])
            active_totals.append(key_length)

    # Combine finished and remaining active batches
    return finished_batches + active_batches


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
        ValueError: If batch_method is invalid or if any sample
            length exceeds batch_token.
    """
    # Check if any single sample exceeds batch_token
    for key in keys:
        key_length = key_to_length[key]
        if key_length > batch_token:
            raise ValueError(
                f"Sample {key} has length {key_length} which "
                f"exceeds batch_token limit {batch_token}"
            )

    if batch_method == "bucket":
        return batchfy_bucket(keys, key_to_length, batch_token)
    elif batch_method == "pack":
        return batchfy_pack(keys, key_to_length, batch_token)
    else:
        raise ValueError(
            f"Invalid batch_method: {batch_method}. " f"Must be 'bucket' or 'pack'."
        )


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
    try:
        import logging

        import torch
        import torch.distributed as dist
    except ImportError:
        # torch not available, return batches as-is
        return batches

    if not torch.cuda.is_available() or not dist.is_initialized():
        return batches

    n_batches = len(batches)
    n_batches_tensor = torch.Tensor([n_batches]).long().cuda()
    n_batches_list = [
        torch.Tensor([0]).long().cuda() for _ in range(dist.get_world_size())
    ]
    dist.all_gather(n_batches_list, n_batches_tensor)
    tgt_n_batches = max([t.cpu().item() for t in n_batches_list])

    if tgt_n_batches > n_batches:
        batches = batches + batches[-(tgt_n_batches - n_batches) :]
        logging.info("Synchronize sharded dataset across all process")
        logging.info(f"#Batches: {n_batches} -> {tgt_n_batches}")

    return batches
