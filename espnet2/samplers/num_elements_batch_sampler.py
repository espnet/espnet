from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class NumElementsBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        assert check_argument_types()
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
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

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

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")

        if padding:
            for d, s in zip(utt2shapes, shape_files):
                # shape: (Length, dim1, dim2, ...)
                if not all(tuple(d[k][1:]) == tuple(d[keys[0]][1:]) for k in keys):
                    raise RuntimeError(
                        "If padding=True, the feature dimension must be unified: {s}",
                    )
            # If padding case, the feat-dim must be same over whole corpus,
            # therefore the first sample is referred
            feat_dims = [np.prod(d[keys[0]][1:]) for d in utt2shapes]
        else:
            feat_dims = None

        # Decide batch-sizes
        start = 0
        batch_sizes = []
        bs = 1
        while True:
            # shape: (Length, dim1, dim2, ...)
            if padding:
                max_lengths = [
                    max(d[keys[i]][0] for i in range(start, start + bs))
                    for d in utt2shapes
                ]
                bins = sum(bs * lg * d for lg, d in zip(max_lengths, feat_dims))
            else:
                bins = sum(
                    np.prod(d[keys[i]])
                    for i in range(start, start + bs)
                    for d in utt2shapes
                )

            if bins > batch_bins and bs >= min_batch_size:
                batch_sizes.append(bs)
                start += bs
                bs = 1
            else:
                bs += 1
            if start >= len(keys):
                break

            if start + bs > len(keys):
                if not self.drop_last or len(batch_sizes) == 0:
                    batch_sizes.append(len(keys) - start)
                break

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        # If the last batch-size is smaller than minimum batch_size,
        # the samples are redistributed to the other mini-batches
        if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 1] += 1

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

        # Set mini-batch
        self.batch_list = []
        start = 0
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
            f"batch_bins={self.batch_bins}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
