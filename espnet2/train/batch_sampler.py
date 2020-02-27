from abc import ABC
from abc import abstractmethod
import logging
from typing import Iterator
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from torch.utils.data import Sampler
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.utils.fileio import load_num_sequence_text
from espnet2.utils.fileio import read_2column_text


def build_batch_sampler(
    type: str,
    batch_size: int,
    shape_files: Union[Tuple[str, ...], List[str]],
    max_lengths: Sequence[int] = (),
    sort_in_batch: str = "descending",
    sort_batch: str = "ascending",
    drop_last: bool = False,
    min_batch_size: int = 1,
) -> "AbsSampler":
    """Helper function to instantiate BatchSampler

    Args:
        type: mini-batch type. "constant", "seq", "bin", or, "frame"
        batch_size: The mini-batch size
        shape_files: Text files describing the length and dimension
            of each features. e.g. uttA 1330,80
        max_lengths: Used for "seq" mode
        sort_in_batch:
        sort_batch:
        drop_last:
        min_batch_size: Used for "seq" mode
    """
    assert check_argument_types()

    if type == "const_no_sort":
        retval = ConstantBatchSampler(
            batch_size=batch_size, key_file=shape_files[0], drop_last=drop_last
        )

    elif type == "const":
        retval = ConstantSortedBatchSampler(
            batch_size=batch_size,
            shape_file=shape_files[0],
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
        )

    elif type == "seq":
        if len(max_lengths) != len(shape_files):
            raise ValueError(
                f"The number of max_lengths must be equal to "
                f"the number of shape_files: "
                f"{len(max_lengths)} != {len(shape_files)}"
            )
        retval = SequenceBatchSampler(
            batch_size=batch_size,
            shape_files=shape_files,
            max_lengths=max_lengths,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            min_batch_size=min_batch_size,
        )

    elif type == "bin":
        raise NotImplementedError

    elif type == "frame":
        raise NotImplementedError

    else:
        raise RuntimeError(f"Not supported: {type}")
    assert check_return_type(retval)
    return retval


class AbsSampler(Sampler, ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        raise NotImplementedError


class ConstantSortedBatchSampler(AbsSampler):
    """BatchSampler with sorted samples by length.

    Args:
        batch_size:
        shape_file:
        sort_in_batch: 'descending', 'ascending' or None.
        sort_batch:
    """

    def __init__(
        self,
        batch_size: int,
        shape_file: str,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
    ):
        assert check_argument_types()
        assert batch_size > 0
        self.batch_size = batch_size
        self.shape_file = shape_file
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shape = load_num_sequence_text(shape_file, loader_type="csv_int")
        if sort_in_batch == "descending":
            # Sort samples in descending order (required by RNN)
            keys = sorted(utt2shape, key=lambda k: -utt2shape[k][0])
        elif sort_in_batch == "ascending":
            # Sort samples in ascending order
            keys = sorted(utt2shape, key=lambda k: utt2shape[k][0])
        else:
            raise ValueError(
                f"sort_in_batch must be either one of "
                f"ascending, descending, or None: {sort_in_batch}"
            )
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_file}")

        # Apply max(, 1) to avoid 0-batches
        N = max(len(keys) // batch_size, 1)
        if not self.drop_last:
            # Split keys evenly as possible as. Note that If N != 1,
            # the these batches always have size of batch_size at minimum.
            self.batch_list = [
                keys[i * len(keys) // N : (i + 1) * len(keys) // N] for i in range(N)
            ]
        else:
            self.batch_list = [
                tuple(keys[i * batch_size : (i + 1) * batch_size]) for i in range(N)
            ]

        if len(self.batch_list) == 0:
            logging.warning(f"{shape_file} is empty")

        if sort_in_batch != sort_batch:
            if sort_batch not in ("ascending", "descending"):
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )
            self.batch_list.reverse()

        if len(self.batch_list) == 0:
            raise RuntimeError("0 batches")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_file={self.shape_file}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        for batch in self.batch_list:
            yield batch


class ConstantBatchSampler(AbsSampler):
    """BatchSampler with constant batch-size.

    Any sorting is not done in this class,
    so no length information is required,
    This class is convenient for decoding mode,
    or not seq2seq learning e.g. classification.

    Args:
        batch_size:
        key_file:
    """

    def __init__(self, batch_size: int, key_file: str, drop_last: bool = False):
        assert check_argument_types()
        assert batch_size > 0
        self.batch_size = batch_size
        self.key_file = key_file
        self.drop_last = drop_last

        # utt2shape:
        #    uttA <anything is o.k>
        #    uttB <anything is o.k>
        utt2any = read_2column_text(key_file)
        if len(utt2any) == 0:
            logging.warning(f"{key_file} is empty")
        # In this case the, the first column in only used
        keys = list(utt2any)
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {key_file}")

        # Apply max(, 1) to avoid 0-batches
        N = max(len(keys) // batch_size, 1)
        if not self.drop_last:
            # Split keys evenly as possible as. Note that If N != 1,
            # the these batches always have size of batch_size at minimum.
            self.batch_list = [
                keys[i * len(keys) // N : (i + 1) * len(keys) // N] for i in range(N)
            ]
        else:
            self.batch_list = [
                tuple(keys[i * batch_size : (i + 1) * batch_size]) for i in range(N)
            ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"key_file={self.key_file}, "
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        for batch in self.batch_list:
            yield batch


class SequenceBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        max_lengths: Sequence[int],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
    ):
        assert check_argument_types()
        assert batch_size > 0
        self.batch_size = batch_size
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

        # Decide batch-sizes
        start = 0
        batch_sizes = []
        while True:
            k = keys[start]
            factor = max(int(d[k][0] / m) for d, m in zip(utt2shapes, max_lengths))
            bs = max(min_batch_size, int(batch_size / (1 + factor)))
            if self.drop_last and start + bs > len(keys):
                # This if-block avoids 0-batches
                if len(self.batch_list) > 0:
                    break

            bs = min(len(keys) - start, bs)
            batch_sizes.append(bs)
            start += bs
            if start >= len(keys):
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
            f"batch_size={self.batch_size}, "
            f"shape_files={self.shape_files}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        for batch in self.batch_list:
            yield batch
