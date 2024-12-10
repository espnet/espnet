from typing import List, Optional, Sequence, Tuple, Union

from typeguard import typechecked

from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.samplers.folded_batch_sampler import FoldedBatchSampler
from espnet2.samplers.length_batch_sampler import LengthBatchSampler
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler
from espnet2.samplers.sorted_batch_sampler import SortedBatchSampler
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler

BATCH_TYPES = dict(
    unsorted="UnsortedBatchSampler has nothing in particular feature and "
    "just creates mini-batches which has constant batch_size. "
    "This sampler doesn't require any length "
    "information for each feature. "
    "'key_file' is just a text file which describes each sample name."
    "\n\n"
    "    utterance_id_a\n"
    "    utterance_id_b\n"
    "    utterance_id_c\n"
    "\n"
    "The fist column is referred, so 'shape file' can be used, too.\n\n"
    "    utterance_id_a 100,80\n"
    "    utterance_id_b 400,80\n"
    "    utterance_id_c 512,80\n",
    sorted="SortedBatchSampler sorts samples by the length of the first input "
    " in order to make each sample in a mini-batch has close length. "
    "This sampler requires a text file which describes the length for each sample "
    "\n\n"
    "    utterance_id_a 1000\n"
    "    utterance_id_b 1453\n"
    "    utterance_id_c 1241\n"
    "\n"
    "The first element of feature dimensions is referred, "
    "so 'shape_file' can be also used.\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",
    folded="FoldedBatchSampler supports variable batch_size. "
    "The batch_size is decided by\n"
    "    batch_size = base_batch_size // (L // fold_length)\n"
    "L is referred to the largest length of samples in the mini-batch. "
    "This samples requires length information as same as SortedBatchSampler\n",
    length="LengthBatchSampler supports variable batch_size. "
    "This sampler makes mini-batches which have same number of 'bins' as possible "
    "counting by the total lengths of each feature in the mini-batch. "
    "This sampler requires a text file which describes the length for each sample. "
    "\n\n"
    "    utterance_id_a 1000\n"
    "    utterance_id_b 1453\n"
    "    utterance_id_c 1241\n"
    "\n"
    "The first element of feature dimensions is referred, "
    "so 'shape_file' can be also used.\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",
    numel="NumElementsBatchSampler supports variable batch_size. "
    "Just like LengthBatchSampler, this sampler makes mini-batches"
    " which have same number of 'bins' as possible "
    "counting by the total number of elements of each feature "
    "instead of the length. "
    "Thus this sampler requires the full information of the dimension of the features. "
    "\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",
)


@typechecked
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
    utt2category_file: Optional[str] = None,
) -> AbsSampler:
    """
        Helper function to instantiate various types of BatchSampler.

    This function creates and returns an instance of a specified batch sampler
    based on the provided parameters. It supports multiple sampler types, each
    designed for different batching strategies.

    Attributes:
        type (str): The mini-batch type. Options include "unsorted",
            "sorted", "folded", "numel", "length", or "catbel".
        batch_size (int): The mini-batch size. Used for "unsorted",
            "sorted", "folded", and "catbel" modes.
        batch_bins (int): Used for "numel" mode.
        shape_files (Union[Tuple[str, ...], List[str]]): Text files describing
            the length and dimension of each feature, e.g., "uttA 1330,80".
        sort_in_batch (str): Sorting order for samples within each batch.
        sort_batch (str): Sorting order for batches.
        drop_last (bool): Whether to drop the last incomplete batch.
        min_batch_size (int): Minimum batch size used for "numel" or "folded" mode.
        fold_lengths (Sequence[int]): Used for "folded" mode to specify fold lengths.
        padding (bool): Whether sequences are input as a padded tensor (used for
            "numel" mode).
        utt2category_file (Optional[str]): Optional file to categorize utterances.

    Args:
        type: Mini-batch type. Options: "unsorted", "sorted", "folded",
            "numel", "length", or "catbel".
        batch_size: The mini-batch size.
        batch_bins: Number of bins used for "numel" mode.
        shape_files: Text files describing feature dimensions.
        sort_in_batch: Sorting order for samples in each batch.
        sort_batch: Sorting order for batches.
        drop_last: Whether to drop the last incomplete batch.
        min_batch_size: Minimum batch size for "numel" or "folded" mode.
        fold_lengths: Lengths for folding (used in "folded" mode).
        padding: Whether to pad sequences for "numel" mode.
        utt2category_file: Optional file for categorizing utterances.

    Returns:
        AbsSampler: An instance of a batch sampler.

    Raises:
        ValueError: If no shape files are provided or if the number of
            fold_lengths does not match the number of shape_files.

    Examples:
        # Create an unsorted batch sampler
        sampler = build_batch_sampler(
            type="unsorted",
            batch_size=32,
            batch_bins=0,
            shape_files=["shapes.txt"],
        )

        # Create a sorted batch sampler
        sampler = build_batch_sampler(
            type="sorted",
            batch_size=32,
            batch_bins=0,
            shape_files=["shapes.txt"],
            sort_in_batch="ascending",
            sort_batch="descending",
        )

        # Create a folded batch sampler
        sampler = build_batch_sampler(
            type="folded",
            batch_size=64,
            batch_bins=0,
            shape_files=["shapes1.txt", "shapes2.txt"],
            fold_lengths=[10, 20],
        )
    """
    if len(shape_files) == 0:
        raise ValueError("No shape file are given")

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
            utt2category_file=utt2category_file,
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
        retval = LengthBatchSampler(
            batch_bins=batch_bins,
            shape_files=shape_files,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            padding=padding,
            min_batch_size=min_batch_size,
        )

    else:
        raise ValueError(f"Not supported: {type}")
    return retval
