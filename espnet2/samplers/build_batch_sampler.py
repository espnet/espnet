from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from typeguard import typechecked

from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.samplers.category_balanced_sampler import CategoryBalancedSampler
from espnet2.samplers.category_power_sampler import (
    CategoryDatasetPowerSampler,
    CategoryPowerSampler,
)
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

CATEGORY_BATCH_TYPES = dict(
    catbel="CategoryBalancedSampler keeps equally distributed categories (classes)"
    " within each minibatch. If the batch_size is smaller than the number of classes, "
    "all samples in the minibatch will belong to different classes. "
    "This sampler requires a 'category2utt' file which maps each category to "
    "utterance IDs. "
    "\n\n"
    "    category_a utterance_id_a1 utterance_id_a2\n"
    "    category_b utterance_id_b1 utterance_id_b2\n"
    "    category_c utterance_id_c1 utterance_id_c2\n",
    catpow="CategoryPowerSampler constructs mini-batches by balancing samples across "
    "categories using a power-law distribution to control sampling frequency. "
    "The sampling probability is P(x) = (n_l / N)^β * (1 / k_l) where β is the "
    "upsampling factor, n_l is the total duration of category l, N is total duration "
    "of all categories, and k_l is the number of utterances in category l. "
    "Batches are constructed based on 'batch_bins' similar to LengthBatchSampler. "
    "This sampler requires a 'category2utt' file which maps each category to "
    "utterance IDs. "
    "\n\n"
    "    category_a utterance_id_a1 utterance_id_a2\n"
    "    category_b utterance_id_b1 utterance_id_b2\n"
    "    category_c utterance_id_c1 utterance_id_c2\n",
    catpow_balance_dataset="CategoryDatasetPowerSampler performs hierarchical "
    "sampling for multi-category, multi-dataset training where both category "
    "imbalance and dataset imbalance exist. It first balances categories within "
    "each dataset using P(l | d) ∝ (n_ld / N_d)^β_L, then balances datasets "
    "themselves using P(d) ∝ (N_d / M)^β_D. The final sampling probability is "
    "P(x) = P(d) × P(l | d) × P(x | l, d), where P(x | l, d) = 1 / k_ld. This "
    "sampler is particularly useful when combining heterogeneous datasets with "
    "highly imbalanced dataset size and category distributions. Requires "
    "'category2utt', 'dataset2utt', and 'utt2dataset' files. "
    "\n\n"
    "Required file formats:\n"
    "category2utt_file:\n"
    "    category_a utterance_id_a1 utterance_id_a2\n"
    "    category_b utterance_id_b1 utterance_id_b2\n"
    "\n"
    "dataset2utt_file:\n"
    "    dataset1 utterance_id_a1 utterance_id_b2\n"
    "    dataset2 utterance_id_b1 utterance_id_a2\n"
    "\n"
    "utt2dataset_file:\n"
    "    utterance_id_a1 dataset1\n"
    "    utterance_id_a2 dataset2\n"
    "    utterance_id_b1 dataset2\n"
    "    utterance_id_b2 dataset1\n",
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
    """Helper function to instantiate BatchSampler.

    Args:
        type: mini-batch type. "unsorted", "sorted", "folded", "numel",
            "length", or "catbel"
        batch_size: The mini-batch size. Used for "unsorted", "sorted",
            "folded", "catbel" mode
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
        raise ValueError(
            f"Not supported: {type}. "
            "Please specify batch_type in unsorted, sorted, folded, numel, "
            "length."
        )
    return retval


@typechecked
def build_category_batch_sampler(
    type: str,
    batch_size: Optional[int] = None,
    batch_bins: Optional[int] = None,
    shape_files: Optional[Union[Tuple[str, ...], List[str]]] = None,
    min_batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    upsampling_factor: Optional[float] = None,
    category_upsampling_factor: Optional[float] = None,
    dataset_upsampling_factor: Optional[float] = None,
    dataset_scaling_factor: Optional[float] = None,
    drop_last: bool = False,
    category2utt_file: Optional[str] = None,
    dataset2utt_parent_dir: Optional[str] = None,
    epoch: int = 1,
    num_batches: Optional[int] = None,
    distributed: bool = False,
) -> Tuple[AbsSampler, dict]:

    if type == "catbel":
        sampler_args = dict(
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            drop_last=drop_last,
            category2utt_file=category2utt_file,
            epoch=epoch,
            num_batches=num_batches,
            distributed=distributed,
        )
        batch_sampler = CategoryBalancedSampler(**sampler_args)
    elif type == "catpow":
        sampler_args = dict(
            batch_bins=batch_bins,
            shape_files=shape_files,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            upsampling_factor=upsampling_factor,
            dataset_scaling_factor=dataset_scaling_factor,
            drop_last=drop_last,
            category2utt_file=category2utt_file,
            epoch=epoch,
            num_batches=num_batches,
            distributed=distributed,
        )
        batch_sampler = CategoryPowerSampler(**sampler_args)
    elif type == "catpow_balance_dataset":
        if Path(dataset2utt_parent_dir, "dataset2utt").exists():
            dataset2utt_file = str(Path(dataset2utt_parent_dir, "dataset2utt"))
        else:
            dataset2utt_file = None
            raise ValueError(
                f"dataset2utt mandatory for catpow_balance_dataset batch sampler, "
                f"but not found {dataset2utt_file}. To create a dataset2utt "
                "file, please refer to the script in "
                "`egs2/geolid/lid1/local/create_utt2dataset_dataset2utt.sh`"
            )

        if Path(dataset2utt_parent_dir, "utt2dataset").exists():
            utt2dataset_file = str(Path(dataset2utt_parent_dir, "utt2dataset"))
        else:
            utt2dataset_file = None
            raise ValueError(
                f"utt2dataset mandatory for catpow_balance_dataset batch sampler, "
                f"but not found {utt2dataset_file}. To create a dataset2utt "
                "file, please refer to the script in "
                "`egs2/geolid/lid1/local/create_utt2dataset_dataset2utt.sh`"
            )
        sampler_args = dict(
            batch_bins=batch_bins,
            shape_files=shape_files,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            category_upsampling_factor=category_upsampling_factor,
            dataset_upsampling_factor=dataset_upsampling_factor,
            dataset_scaling_factor=dataset_scaling_factor,
            drop_last=drop_last,
            category2utt_file=category2utt_file,
            dataset2utt_file=dataset2utt_file,
            utt2dataset_file=utt2dataset_file,
            epoch=epoch,
            num_batches=num_batches,
            distributed=distributed,
        )
        batch_sampler = CategoryDatasetPowerSampler(**sampler_args)
    else:
        raise ValueError(
            f"batch_type={type} is not supported when iterator_type=category."
            f"Please specify batch_type in {CATEGORY_BATCH_TYPES.keys()}."
        )

    return batch_sampler, sampler_args
