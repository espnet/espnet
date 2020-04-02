from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.samplers.folded_batch_sampler import FoldedBatchSampler
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler
from espnet2.samplers.sorted_batch_sampler import SortedBatchSampler
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler


def build_batch_sampler(
    type: str,
    batch_size: int,
    batch_bins: int,
    shape_files: Union[Tuple[str, ...], List[str]],
    sort_in_batch: str = "descending",
    sort_batch: str = "ascending",
    drop_last: bool = False,
    min_batch_size: int = 1,
    fold_lengths: Sequence[int] = (),
    padding: bool = True,
) -> AbsSampler:
    """Helper function to instantiate BatchSampler.

    Args:
        type: mini-batch type. "unsorted", "sorted", "folded", "numel", or, "length"
        batch_size: The mini-batch size. Used for "unsorted", "sorted", "folded" mode
        batch_bins: Used for "numel" model
        shape_files: Text files describing the length and dimension
            of each features. e.g. uttA 1330,80
        sort_in_batch:
        sort_batch:
        drop_last:
        min_batch_size:  Used for "numel" or "folded" mode
        fold_lengths: Used for "folded" mode
        padding: Whether sequences are input as a padded tensor or not.
            used for "numel" mode
    """
    assert check_argument_types()

    if type == "unsorted":
        retval = UnsortedBatchSampler(
            batch_size=batch_size, key_file=shape_files[0], drop_last=drop_last
        )

    elif type == "sorted":
        retval = SortedBatchSampler(
            batch_size=batch_size,
            shape_file=shape_files[0],
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
        )

    elif type == "folded":
        if len(fold_lengths) != len(shape_files):
            raise ValueError(
                f"The number of fold_lengths must be equal to "
                f"the number of shape_files: "
                f"{len(fold_lengths)} != {len(shape_files)}"
            )
        retval = FoldedBatchSampler(
            batch_size=batch_size,
            shape_files=shape_files,
            fold_lengths=fold_lengths,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            min_batch_size=min_batch_size,
        )

    elif type == "numel":
        retval = NumElementsBatchSampler(
            batch_bins=batch_bins,
            shape_files=shape_files,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            padding=padding,
            min_batch_size=min_batch_size,
        )

    elif type == "length":
        raise NotImplementedError

    else:
        raise ValueError(f"Not supported: {type}")
    assert check_return_type(retval)
    return retval
