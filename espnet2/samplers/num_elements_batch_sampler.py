from typing import Iterator, List, Tuple, Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class NumElementsBatchSampler(AbsSampler):
    """
        NumElementsBatchSampler is a batch sampler that creates mini-batches of data
    based on the number of elements (bins) specified. This sampler is designed to
    work with variable-length sequences, ensuring that each batch contains a
    specified maximum number of bins while maintaining a minimum batch size.
    The samples can be sorted in various orders within each batch and across
    batches.

    Attributes:
        batch_bins (int): The maximum number of bins allowed per batch.
        shape_files (Union[Tuple[str, ...], List[str]]): List of paths to files
            containing sequence lengths.
        sort_in_batch (str): The order in which to sort elements within each
            batch; options are "ascending" or "descending".
        sort_batch (str): The order in which to sort batches; options are
            "ascending" or "descending".
        drop_last (bool): Whether to drop the last incomplete batch if it has
            fewer than `min_batch_size` elements.
        batch_list (List[Tuple[str, ...]]): The final list of batches created
            by the sampler.

    Args:
        batch_bins (int): Maximum number of bins in each batch.
        shape_files (Union[Tuple[str, ...], List[str]]): List of shape files
            containing sequence lengths.
        min_batch_size (int, optional): Minimum number of samples in a batch.
            Defaults to 1.
        sort_in_batch (str, optional): Sorting order for elements in a batch.
            Can be "ascending" or "descending". Defaults to "descending".
        sort_batch (str, optional): Sorting order for batches. Can be
            "ascending" or "descending". Defaults to "ascending".
        drop_last (bool, optional): If True, drop the last batch if it is
            smaller than `min_batch_size`. Defaults to False.
        padding (bool, optional): If True, ensures all features have the same
            dimension across the corpus. Defaults to True.

    Returns:
        Iterator[Tuple[str, ...]]: An iterator over the batches.

    Raises:
        ValueError: If `sort_batch` or `sort_in_batch` is not "ascending" or
        "descending".
        RuntimeError: If the keys in the shape files do not match or if no
        batches can be created.

    Examples:
        >>> sampler = NumElementsBatchSampler(batch_bins=10,
        ...                                    shape_files=["file1.csv", "file2.csv"])
        >>> for batch in sampler:
        ...     print(batch)

    Note:
        This sampler is useful for training models with variable-length input
        sequences, such as in speech or language processing tasks.
    """

    @typechecked
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

            if bins > batch_bins and len(current_batch_keys) >= min_batch_size:
                batch_sizes.append(len(current_batch_keys))
                current_batch_keys = []
        else:
            if len(current_batch_keys) != 0 and (
                not self.drop_last or len(batch_sizes) == 0
            ):
                batch_sizes.append(len(current_batch_keys))

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
