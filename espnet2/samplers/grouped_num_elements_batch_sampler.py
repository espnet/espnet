from typing import Iterator, List, Tuple, Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class GroupedNumElementsBatchSampler(AbsSampler):
    """
    A variation of numel batchsampler 
    that groups samples by input length.
    Allows each GPU to have a near identical batch size.
    Uses a per-GPU level batch bins.

    Designed for large-scale multi-node training.
    """
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
        self.biggest_batch_size = 0
        self.biggest_batch_keys = None

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
            # If padding case, the feat-dim must be same over whole corpus,
            # therefore the first sample is referred
            feat_dims = [np.prod(d[keys[0]][1:]) for d in utt2shapes]
        else:
            feat_dims = None

        # Decide batch-sizes
        batch_sizes = []
        current_batch_keys = []

        for key in keys:
            current_batch_keys.append(key)
            # shape: (Length, dim1, dim2, ...)
            if padding:
                for d, s in zip(utt2shapes, shape_files):
                    if tuple(d[key][1:]) != tuple(d[keys[0]][1:]):
                        raise RuntimeError(
                            "If padding=True, the "
                            f"feature dimension must be unified: {s}",
                        )
                bins = sum(
                    len(current_batch_keys) * sh[key][0] * d
                    for sh, d in zip(utt2shapes, feat_dims)
                )
            else:
                bins = sum(
                    np.prod(d[k]) for k in current_batch_keys for d in utt2shapes
                )

            # batch_bins is now gpu-level
            if bins > batch_bins:
                current_batch_keys = current_batch_keys[:-1]
                if padding:
                    for d, s in zip(utt2shapes, shape_files):
                        if tuple(d[key][1:]) != tuple(d[keys[0]][1:]):
                            raise RuntimeError(
                                "If padding=True, the "
                                f"feature dimension must be unified: {s}",
                            )
                    bins = sum(
                        len(current_batch_keys) * sh[key][0] * d
                        for sh, d in zip(utt2shapes, feat_dims)
                    )
                else:
                    bins = sum(
                        np.prod(d[k]) for k in current_batch_keys for d in utt2shapes
                    )
                self.biggest_batch_size = max(bins, self.biggest_batch_size)
                self.biggest_batch_keys = current_batch_keys
                batch_sizes.append(len(current_batch_keys))
                current_batch_keys = [key]
        else:
            if len(current_batch_keys) != 0:
                batch_sizes.append(len(current_batch_keys))

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        if len(batch_sizes) % min_batch_size != 0:
            whole_batches = len(batch_sizes) // min_batch_size

            # we have x amount of left over batches
            leftovers = batch_sizes[whole_batches*min_batch_size:]

            # first get total number of samples
            total_samples = sum(leftovers)

            samples_per_batch = total_samples // min_batch_size

            if samples_per_batch > 0: # dont distribute if we dont have much data left

                old_total_samples = sum(batch_sizes)

                # we could over allocate past the batch size if the distribution is uneven
                # try to split it more if possible
                for i in range(2,8):
                    if samples_per_batch % i == 0: 
                        new_batch_sizes = [samples_per_batch // i] * (min_batch_size * i)
                        break
                else:
                    new_batch_sizes = [samples_per_batch] * min_batch_size

                batch_sizes = batch_sizes[0:whole_batches*min_batch_size] + new_batch_sizes
                assert len(batch_sizes) % min_batch_size == 0
                assert old_total_samples >= sum(batch_sizes)
            else:
                batch_sizes = batch_sizes[0:whole_batches*min_batch_size]

        # Set mini-batch
        self.batch_list = []
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys = []
        for key in keys:
            minibatch_keys.append(key)
            if len(minibatch_keys) == bs:
                if sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending"
                        f" or descending: {sort_in_batch}"
                    )

                self.batch_list.append(tuple(minibatch_keys))
                minibatch_keys = []
                try:
                    bs = next(iter_bs)
                except StopIteration:
                    break

        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )

        new_batch_list = []
        batch_list_grouped = []
        for i, batch in enumerate(self.batch_list):
            if i > min_batch_size - 1 and i % min_batch_size == 0:
                assert len(batch_list_grouped) == min_batch_size
                new_batch_list.append(batch_list_grouped)
                batch_list_grouped = []
            batch_list_grouped.append(batch)
        else:
            if len(batch_list_grouped) > 0:
                assert len(batch_list_grouped) == min_batch_size, f"{len(batch_list_grouped), min_batch_size}"
                new_batch_list.append(batch_list_grouped)
        self.batch_list = new_batch_list
        for batch in self.batch_list:
            assert len(batch) == min_batch_size, f"{len(batch), min_batch_size}"

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
