#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Iterator, List, Tuple, Union
from typeguard import typechecked
from collections import defaultdict

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler

class BucketBatchSampler(AbsSampler):
    """
    Bucket Batch Sampler:
    With the given shape file, form mini-batches of data that have constant
    batch size and batch length. Multiple examples can be concatenated as a
    long sequence and then achieve constant batch length.

    Note: to be compatible with other batch samplers, this sampler only 
    returns list of utterance_ids for each minibatch. The collate_fn will 
    assemble these items into the padded batches.
    """
    @typechecked
    def __init__(
        self,
        batch_bins: int,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        allow_duplication: bool = False,
    ):
        self.batch_bins = batch_bins
        self.batch_size = batch_size
        self.shape_files = shape_files
        self.allow_duplication = allow_duplication

        # (1) load the shape file and sort by length
        assert len(shape_files) == 1, "Only one shape file is allowed"

        utt2shapes = load_num_sequence_text(
            shape_files[0], loader_type="csv_int", allow_duplication=allow_duplication
        )
        keys = sorted(utt2shapes.keys(), key=lambda x: utt2shapes[x], reverse=True)

        # (2) form buckets
        bins = defaultdict(lambda: {'items': [], 'total': 0})
        bin_id_counter = 0
        
        # install this library only when in use
        from sortedcontainers import SortedList
        residuals = SortedList()
        
        # Process each item
        for name in keys:
            length = utt2shapes[name][0]

            if length > 800:
                continue

            # Find the leftmost bin with residual_capacity >= length
            index = residuals.bisect_left((length, 0))
            
            if index < len(residuals):
                # Suitable bin found
                residual_capacity, bin_id = residuals.pop(index)
                bins[bin_id]['items'].append(name)
                bins[bin_id]['total'] += length
                new_residual = residual_capacity - length
                # Insert the updated residual capacity back into the list
                residuals.add((new_residual, bin_id))
            else:
                # No suitable bin found, create a new bin
                bins[bin_id_counter]['items'].append(name)
                bins[bin_id_counter]['total'] += length
                residual_capacity = batch_bins - length
                residuals.add((residual_capacity, bin_id_counter))
                bin_id_counter += 1  # Increment bin ID for the next new bin
        
        bucket_list = [bin_info["items"] for bin_info in bins.values()]

        # (3) split into batches. all buckets are merged into one list for compatibility
        # but they will later be parsed in the collate_fn
        self.batch_list = []
        b_idx = 0
        while b_idx < len(bucket_list):
            max_idx = min(b_idx + batch_size, len(bucket_list))
            this_batch = [n for bucket in bucket_list[b_idx: max_idx] for n in bucket]
            self.batch_list.append(tuple(this_batch))
            b_idx = b_idx + batch_size
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"batch_size={self.batch_size}, "
            f"shape_files={self.shape_files}, "
            f"allow_duplication={self.allow_duplication} "
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)