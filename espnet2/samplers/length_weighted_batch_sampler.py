from typing import Iterator, List, Tuple, Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class WeightedLengthBatchSampler(AbsSampler):
    @typechecked
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        utt2weight_file: str,
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        assert batch_bins > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_bins = batch_bins
        self.shape_files = shape_files
        self.utt2weight_file = utt2weight_file
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last
        self.padding = padding
        self.min_batch_size = min_batch_size
        self.np_seed = 0

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )

        utt2weight = load_num_sequence_text(utt2weight_file, loader_type="text_float")
        if set(first_utt2shape) != set(utt2weight):
            raise RuntimeError(
                f"keys are mismatched between shape_files and {utt2weight_file}"
            )

        self.keys = sorted(utt2weight, key=lambda k: k)
        if len(self.keys) == 0:
            raise RuntimeError(f"0 lines found: {utt2weight_file}")

        self.weights = [utt2weight[k][0] for k in self.keys]
        self.weights = np.array(self.weights) / np.sum(self.weights)

        self.key2shape = {k: first_utt2shape[k] for k in self.keys}
        self.utt2shapes = utt2shapes

        self.batch_list = self.generate_batch_list_()

    def generate_batch_list_(self):
        """Generate a batch list with weighted sampling and length-based batching"""
        # Sample utterances based on weights
        np.random.seed(self.np_seed)
        self.np_seed += 1

        indices = np.random.choice(len(self.keys), len(self.keys), p=self.weights)
        sampled_keys = [self.keys[idx] for idx in indices]

        sorted_keys = sorted(sampled_keys, key=lambda k: self.key2shape[k][0])

        batch_sizes = []
        current_batch_keys = []
        for key in sorted_keys:
            current_batch_keys.append(key)
            if self.padding:
                # bins = bs x max_length
                bins = sum(
                    len(current_batch_keys) * sh[key][0] for sh in self.utt2shapes
                )
            else:
                # bins = sum of lengths
                bins = sum(d[k][0] for k in current_batch_keys for d in self.utt2shapes)

            if (
                bins > self.batch_bins
                and len(current_batch_keys) >= self.min_batch_size
            ):
                batch_sizes.append(len(current_batch_keys))
                current_batch_keys = []
        else:
            if len(current_batch_keys) != 0 and (
                not self.drop_last or len(batch_sizes) == 0
            ):
                batch_sizes.append(len(current_batch_keys))

        if len(batch_sizes) == 0:
            return []

        if len(batch_sizes) > 1 and batch_sizes[-1] < self.min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 1] += 1

        batches = []
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys = []
        for key in sorted_keys:
            minibatch_keys.append(key)
            if len(minibatch_keys) == bs:
                if self.sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif self.sort_in_batch == "ascending":
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending"
                        f" or descending: {self.sort_in_batch}"
                    )
                batches.append(tuple(minibatch_keys))
                minibatch_keys = []
                try:
                    bs = next(iter_bs)
                except StopIteration:
                    break

        if self.sort_batch == "ascending":
            pass
        elif self.sort_batch == "descending":
            batches.reverse()

        return batches

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        self.batch_list = self.generate_batch_list_()
        return iter(self.batch_list)
