from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.samplers.batch_bin_sampler import BatchBinSampler
from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler
from espnet2.samplers.constant_sorted_batch_sampler import ConstantSortedBatchSampler
from espnet2.samplers.sequence_batch_sampler import SequenceBatchSampler


def build_batch_sampler(
    type: str,
    batch_size: int,
    batch_bins: int,
    shape_files: Union[Tuple[str, ...], List[str]],
    fold_lengths: Sequence[int] = (),
    sort_in_batch: str = "descending",
    sort_batch: str = "ascending",
    drop_last: bool = False,
    min_batch_size: int = 1,
    padding: bool = True,
) -> AbsSampler:
    """Helper function to instantiate BatchSampler

    Args:
        type: mini-batch type. "constant", "seq", "bin", or, "frame"
        batch_size: The mini-batch size
        batch_bins:
        shape_files: Text files describing the length and dimension
            of each features. e.g. uttA 1330,80
        fold_lengths: Used for "seq" mode
        sort_in_batch:
        sort_batch:
        drop_last:
        min_batch_size: Used for "seq" or "bin" mode
        padding: Whether sequences are input as a padded tensor or not.
            used for "bin" mode
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
        if len(fold_lengths) != len(shape_files):
            raise ValueError(
                f"The number of fold_lengths must be equal to "
                f"the number of shape_files: "
                f"{len(fold_lengths)} != {len(shape_files)}"
            )
        retval = SequenceBatchSampler(
            batch_size=batch_size,
            shape_files=shape_files,
            fold_lengths=fold_lengths,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            min_batch_size=min_batch_size,
        )

    elif type == "bin":
        retval = BatchBinSampler(
            batch_bins=batch_bins,
            shape_files=shape_files,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            padding=padding,
            min_batch_size=min_batch_size,
        )

    elif type == "frame":
        raise NotImplementedError

    else:
        raise ValueError(f"Not supported: {type}")
    assert check_return_type(retval)
    return retval
