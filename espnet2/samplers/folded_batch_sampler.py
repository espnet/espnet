from typing import Iterator
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import h5py
from typeguard import check_argument_types

from espnet2.samplers.abs_sampler import AbsSampler


class FoldedBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_size: int,
        shape_files: Union[
            Tuple[Union[str, h5py.Group], ...], List[Union[str, h5py.Group]]
        ],
        fold_lengths: Sequence[int],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
    ):
        assert check_argument_types()
        assert batch_size > 0
        if len(fold_lengths) != len(shape_files):
            raise ValueError(
                f"The number of fold_lengths must be equal to "
                f"the number of shape_files: "
                f"{len(fold_lengths)} != {len(shape_files)}"
            )
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_size = batch_size
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes, keys = self._load_shape_files(shape_files, sort=True)
        num_keys = len(keys)

        # Decide batch-sizes
        start = 0
        batch_sizes = []
        while True:
            k = keys[start]
            factor = max(int(d[k][0] / m) for d, m in zip(utt2shapes, fold_lengths))
            bs = max(min_batch_size, int(batch_size / (1 + factor)))
            if self.drop_last and start + bs > num_keys:
                # This if-block avoids 0-batches
                if len(batch_sizes) > 0:
                    break

            bs = min(num_keys - start, bs)
            batch_sizes.append(bs)
            start += bs
            if start >= num_keys:
                break

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        # If the last batch-size is smaller than minimum batch_size,
        # the samples are redistributed to the other mini-batches
        if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 2] += 1

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == num_keys, f"{sum(batch_sizes)} != {num_keys}"

        # Set mini-batch
        start = 0
        self.batch_list = []
        for bs in batch_sizes:
            assert len(keys) >= start + bs, "Bug"
            minibatch_keys = keys[start : start + bs]
            start += bs
            if sort_in_batch == "descending":
                minibatch_keys.reverse()
            elif sort_in_batch == "ascending":
                # Key are already sorted in ascending
                pass
            else:
                raise ValueError(
                    f"sort_in_batch must be ascending or descending: {sort_in_batch}"
                )
            self.batch_list.append(tuple(minibatch_keys))

        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_files={self.shape_files}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
