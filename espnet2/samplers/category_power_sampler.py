import random
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import load_num_sequence_text, read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler


class CategoryPowerSampler(AbsSampler):
    r"""A category-balanced batch sampler with power-law sampling.

    Reference:
        Scaling Speech Technology to 1,000+ Languages
        https://arxiv.org/pdf/2305.13516

    This sampler constructs mini-batches by balancing samples across
    categories (e.g., language IDs), using a power-law distribution
    to control the sampling frequency. Originally developed for language
    identification, it can be applied to any dataset that provides a
    mapping from category (e.g., language) to utterances.

    Sampling Strategy:

    Given:
    - l ∈ {1, 2, ..., L}, the set of category labels
    - n_l: total duration (number of bins) of category l
    - N: total duration (number of bins) of all categories in the dataset
    - β: upsampling factor
    - k_l: the number of utterances in category l

    We define:
    - Category-level sampling probability:
        P(l) = (n_l / N)^β
    - Utterance-level conditional sampling:
        P(x | l) = 1 / k_l
    - Combined sampling probability:
        P(x) = P(l) * P(x | l) = (n_l / N)^β * (1 / k_l)

    Where β ∈ [0, 1] is the `upsampling_factor`:
    - β → 0 emphasizes low-resource categories (strong upsampling)
    - β → 1 approximates uniform sampling over all utterances

    Note:
    - Batches are constructed based on `batch_bins`, similar to
    LengthBatchSampler.
    - Set `batch_type=catpow` in your configuration to use this sampler.

    Args:
        batch_bins: The approximate maximum number of bins (e.g., audio samples)
                    in a batch.
        shape_files: A list or tuple of shape file paths. Only one shape
                     file is supported, but the list format is retained for
                     compatibility with other samplers.
        min_batch_size: Minimum number of utterances in a batch.
        max_batch_size: Maximum number of utterances in a batch (recommended
                        for memory safety).
        upsampling_factor: β in the sampling formula; controls how strongly to
                           upsample low-resource categories.
        dataset_scaling_factor: A multiplier that determines the total number of
                                utterances sampled. Values > 1 simulate more frequent
                                use of low-resource utterances across batches.
                                Must be ≥ 1.
        drop_last: Whether to drop the final batch.
        category2utt_file: Path to a file mapping each category to utterance ID.
        epoch: Random seed is set using the epoch to ensure reproducibility with
               variation across epochs.
    """

    @typechecked
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        upsampling_factor: float = 1.0,
        dataset_scaling_factor: float = 1.2,
        drop_last: bool = False,
        category2utt_file: Optional[str] = None,
        epoch: int = 1,
        **kwargs,
    ):
        assert batch_bins > 0
        assert category2utt_file is not None
        assert dataset_scaling_factor >= 1, "dataset_scaling_factor must >= 1"

        # Set random seed based on epoch to ensure different sampling at each epoch
        self.random_state = random.Random(epoch)
        self.np_random_state = np.random.RandomState(epoch)

        self.batch_bins = batch_bins
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.upsampling_factor = upsampling_factor
        self.dataset_scaling_factor = dataset_scaling_factor
        self.category2utt_file = category2utt_file

        assert len(shape_files) == 1, "Only one shape file is supported"
        utt2sizes = [
            load_num_sequence_text(s, loader_type="text_int") for s in shape_files
        ]  # A list of dict: key is utt id, value is speech size

        # Load category -> list of utterances
        category2utt_raw = read_2columns_text(category2utt_file)
        self.category2utt = {k: v.split(" ") for k, v in category2utt_raw.items()}
        self.categories = list(self.category2utt.keys())

        # 1. Compute n_l (the number of bins in each category) and
        # N (total number of bins in the dataset)
        self.category_bins = {
            cat: sum(utt2sizes[0][utt][0] for utt in utts)
            for cat, utts in self.category2utt.items()
        }
        total_bins = sum(self.category_bins.values())

        # 2. Compute sampling prob of each category: P(l) = (n_l / N)^β
        probs = np.array(
            [
                (self.category_bins[cat] / total_bins) ** upsampling_factor
                for cat in self.categories
            ]
        )
        probs /= probs.sum()  # normalize
        self.category_probs = probs

        # 3. Flatten all utts by category
        self.all_utts_by_category = defaultdict(list)
        for cat, utts in self.category2utt.items():
            self.all_utts_by_category[cat].extend(utts)
            # Shuffle utterances within each category to ensure
            # P(x | l) = 1 / k_l (uniform sampling within category)
            self.random_state.shuffle(self.all_utts_by_category[cat])

        # 4. Estimate the total number of utterances after upsampling the whole dataset
        # Motivation: Upsampling low-resource categories may increase the
        # total number of bins beyond the original dataset size. To reflect this,
        # we scale the total bins by `dataset_scaling_factor`, then divide by
        # average utterance size to get the total number of utterances to sample.
        assert dataset_scaling_factor >= 1, "dataset_scaling_factor must >= 1"
        scaling_bins = int(total_bins * dataset_scaling_factor)
        utt_avg_size = np.mean([utt2sizes[0][utt][0] for utt in utt2sizes[0].keys()])
        total_num_samples = int(scaling_bins / utt_avg_size)

        # 5. Sample utterances according to category_probs.
        # Use cat_ptr as a pointer for each category to cycle through its utterances,
        # to avoid excessive repetition of the same utterance in a batch.
        cat_ptr = {cat: 0 for cat in self.categories}
        sampled_utts = []
        for _ in range(total_num_samples):
            # P(l)
            cat = self.np_random_state.choice(self.categories, p=self.category_probs)
            # The index may wrap around to the start of the utterance list,
            # but this does not cause excessive repetition within a batch,
            # since batches are constructed sequentially and utterances are
            # shuffled per epoch.
            idx = cat_ptr[cat] % len(self.all_utts_by_category[cat])
            utt = self.all_utts_by_category[cat][idx]
            cat_ptr[cat] += 1
            sampled_utts.append(utt)

        # 6. Patch sampled utterances into batches
        self.batch_list = []
        current_batch = []
        current_batch_bins = 0
        for utt in sampled_utts:
            utt_size = utt2sizes[0][utt][0]

            current_batch.append(utt)
            current_batch_bins += utt_size

            if (
                current_batch_bins > self.batch_bins
                and len(current_batch) >= self.min_batch_size
            ) or (
                self.max_batch_size is not None
                and len(current_batch) >= self.max_batch_size
            ):
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_bins = 0

        # 7. If the last batch is not empty, append it to the batch list
        if not self.drop_last and len(current_batch) >= self.min_batch_size:
            self.batch_list.append(current_batch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"min_batch_size={self.min_batch_size}, "
            f"max_batch_size={self.max_batch_size}, "
            f"upsampling_factor={self.upsampling_factor}, "
            f"dataset_scaling_factor={self.dataset_scaling_factor}, "
            f"drop_last={self.drop_last}, "
            f"category2utt_file={self.category2utt_file}"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)


class CategoryDatasetPowerSampler(AbsSampler):
    r"""A category- and dataset-balanced batch sampler with power-law sampling.

    Reference:
        Scaling Speech Technology to 1,000+ Languages
        https://arxiv.org/pdf/2305.13516

    This sampler is designed for multi-category, multi-dataset
    training where both category imbalance and dataset imbalance
    exist. It performs hierarchical sampling: (1) balancing categories
    (e.g., languages) within each dataset, and (2) balancing datasets
    themselves.

    Sampling Strategy:

    Let:
    - d ∈ {1, 2, ..., D} denote the dataset index
    - l ∈ {1, 2, ..., L_d} denote the category index in dataset d
    - n_ld: total duration (number of bins) of category l in dataset d
    - k_ld: the number of utterances in category l in dataset d
    - N_d = ∑_l n_ld: total duration (number of bins) of all categories
                      in dataset d
    - M = ∑_d N_d: total duration (number of bins) of all categories across
                all datasets

    Step 1 — Category-level sampling within each dataset:
        P(l | d) ∝ (n_ld / N_d)^β_L

    where β_L (`category_upsampling_factor`) controls how strongly to upsample
    low-resource languages within each dataset. The normalized probability becomes:
        P(l | d) = [(n_ld / N_d)^β_L] / ∑_l'[(n_l'd / N_d)^β_L]

    Step 2 — Dataset-level sampling based on resampled language distributions:

    For each dataset d, the resampled number of bins for category l is:
      n_ld' = N_d × P(l | d)
    Since the category probabilities sum to 1 within each dataset
    (∑_l P(l | d) = 1), the total resampled bins (N_d') for dataset d is:
      N_d' = ∑_l n_ld' = N_d

    The probability of sampling dataset d is then:
      P(d) = [(N_d / M)^β_D] / ∑_d[(N_d / M)^β_D]
    where:
    - β_D is `dataset_upsampling_factor`

    Final utterance sampling probability:
        P(x) = P(d) × P(l | d) × P(x | l, d), where P(x | l, d) = 1 / k_ld

    Note:
    - Batches are constructed based on `batch_bins`, similar to
    LengthBatchSampler.
    - Set `batch_type=catpow_balance_dataset` to enable this sampler.
    - This sampler is particularly useful when combining heterogeneous
    datasets  (e.g., FLEURS + VoxLingua107 + BABEL) with highly imbalanced
    language and size distributions.

    Args:
        batch_bins: The approximate maximum number of bins (e.g., audio samples)
                    in a batch.
        shape_files: A list or tuple of shape file paths. Only one shape
                     file is supported, but the list format is retained for
                     compatibility with other samplers.
        min_batch_size: Minimum number of utterances in a batch.
        max_batch_size: Maximum number of utterances in a batch (recommended
                        for memory safety).
        category_upsampling_factor: β_L in the formula; controls per-dataset
                                    category balancing.
        dataset_upsampling_factor: β_D in the formula; controls balancing between
                                   datasets.
        dataset_scaling_factor: A multiplier that determines the total number of
                                utterances sampled. Values > 1 simulate more frequent
                                use of low-resource utterances across batches.
                                Must be ≥ 1.
        drop_last: Whether to drop the final batch.
        category2utt_file: Path to a file mapping each category to utterance ID.
        dataset2utt_file: Path to a file mapping each dataset to utterance ID.
        utt2dataset_file: Path to a file mapping each utterance ID to its
                          corresponding dataset label.
        epoch: Random seed is set using the epoch to ensure reproducibility with
               variation across epochs.
    """

    @typechecked
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        category_upsampling_factor: float = 1.0,  # β_L
        dataset_upsampling_factor: float = 1.0,  # β_D
        dataset_scaling_factor: float = 1.2,
        drop_last: bool = False,
        category2utt_file: Optional[str] = None,
        dataset2utt_file: Optional[str] = None,
        utt2dataset_file: Optional[str] = None,
        epoch: int = 1,
        **kwargs,
    ):
        assert batch_bins > 0
        assert category2utt_file is not None
        assert dataset2utt_file is not None
        assert utt2dataset_file is not None
        assert dataset_scaling_factor >= 1, "dataset_scaling_factor must >= 1"

        # Set random seed as epoch
        self.random_state = random.Random(epoch)
        self.np_random_state = np.random.RandomState(epoch)

        self.batch_bins = batch_bins
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.category_upsampling_factor = category_upsampling_factor
        self.dataset_upsampling_factor = dataset_upsampling_factor
        self.dataset_scaling_factor = dataset_scaling_factor
        self.category2utt_file = category2utt_file
        self.dataset2utt_file = dataset2utt_file

        assert len(shape_files) == 1, "only one shape file is supported"
        utt2sizes = [
            load_num_sequence_text(s, loader_type="text_int") for s in shape_files
        ]

        # Load mappings
        category2utt_raw = read_2columns_text(category2utt_file)
        self.category2utt = {k: v.split(" ") for k, v in category2utt_raw.items()}

        dataset2utt_raw = read_2columns_text(dataset2utt_file)
        self.dataset2utt = {k: v.split(" ") for k, v in dataset2utt_raw.items()}

        utt2dataset_raw = read_2columns_text(utt2dataset_file)
        self.utt2dataset = {k: v for k, v in utt2dataset_raw.items()}

        self.categories = list(self.category2utt.keys())
        self.datasets = list(self.dataset2utt.keys())

        # Step 1: Category sampling within each dataset
        self.dataset_category_probs = {}
        self.dataset_resampled_bins = {}

        for dataset in self.datasets:
            # Get utterances in this dataset
            dataset_utts = set(self.dataset2utt[dataset])

            # Compute n_{l,d} for each category this dataset
            category_bins_in_dataset = {}
            for category in self.categories:
                category_utts = set(self.category2utt[category])
                # Intersection: utterances that are both in this category
                # and this dataset
                common_utts = category_utts.intersection(dataset_utts)
                if common_utts:
                    category_bins_in_dataset[category] = sum(
                        utt2sizes[0][utt][0] for utt in common_utts
                    )

            if not category_bins_in_dataset:
                continue

            # Compute N_d (total bins in dataset d)
            total_bins_in_dataset = sum(category_bins_in_dataset.values())

            # Compute P(l | d) for each category in this dataset
            category_probs = {}
            prob_values = []
            categories_in_dataset = []

            for category, bins in category_bins_in_dataset.items():
                prob = (bins / total_bins_in_dataset) ** self.category_upsampling_factor
                prob_values.append(prob)
                categories_in_dataset.append(category)

            # Normalize probabilities
            prob_values = np.array(prob_values)
            prob_values /= prob_values.sum()

            for i, category in enumerate(categories_in_dataset):
                category_probs[category] = prob_values[i]

            self.dataset_category_probs[dataset] = category_probs

            # Compute resampled bins: n'_{l,d} = N_d * P(l | d)
            resampled_bins = {}
            for category, prob in category_probs.items():
                resampled_bins[category] = total_bins_in_dataset * prob

            # N_d' = sum(n'_{l,d} for all l) = N_d
            self.dataset_resampled_bins[dataset] = sum(resampled_bins.values())

        # Step 2: Dataset sampling using resampled data
        # Compute M = sum(N_d (N_d' = N_d) for all d)
        total_resampled_bins = sum(self.dataset_resampled_bins.values())

        # Compute P(d) for each dataset
        dataset_probs = []
        for dataset in self.datasets:
            if dataset in self.dataset_resampled_bins:
                prob = (
                    self.dataset_resampled_bins[dataset] / total_resampled_bins
                ) ** self.dataset_upsampling_factor
                dataset_probs.append(prob)
            else:
                dataset_probs.append(0.0)

        # Normalize dataset probabilities
        dataset_probs = np.array(dataset_probs)
        if dataset_probs.sum() > 0:
            dataset_probs /= dataset_probs.sum()
        self.dataset_probs = dataset_probs

        # Prepare utterance lists by dataset and category
        self.dataset_category_utts = defaultdict(lambda: defaultdict(list))
        for dataset in self.datasets:
            if dataset not in self.dataset_category_probs:
                continue
            dataset_utts = set(self.dataset2utt[dataset])
            for category in self.dataset_category_probs[dataset].keys():
                category_utts = set(self.category2utt[category])
                common_utts = list(category_utts.intersection(dataset_utts))
                self.random_state.shuffle(common_utts)  # P(x | l, d) = 1 / k_ld
                self.dataset_category_utts[dataset][category] = common_utts

        # Estimate total number of samples after applying dataset scaling.
        # Motivation: Upsampling low-resource datasets and categories may
        # increase the total number of bins beyond the original dataset size.
        # To reflect this, we scale the total bins by `dataset_scaling_factor`,
        # then divide by average utterance size to get the total number of
        # utterances to sample.
        scaling_bins = int(total_resampled_bins * dataset_scaling_factor)
        utt_avg_size = np.mean([utt2sizes[0][utt][0] for utt in utt2sizes[0].keys()])
        total_num_samples = int(scaling_bins / utt_avg_size)

        # Sample utterances using two-step process
        dataset_category_ptrs = defaultdict(lambda: defaultdict(int))
        sampled_utts = []

        for _ in range(total_num_samples):
            # Step 1: Sample dataset d according to P(d)
            dataset = self.np_random_state.choice(self.datasets, p=self.dataset_probs)

            if dataset not in self.dataset_category_probs:
                continue

            # Step 2: Sample category l according to P(l | d)
            categories_in_dataset = list(self.dataset_category_probs[dataset].keys())
            category_probs_in_dataset = [
                self.dataset_category_probs[dataset][cat]
                for cat in categories_in_dataset
            ]

            if not categories_in_dataset:
                continue

            category = self.np_random_state.choice(
                categories_in_dataset, p=category_probs_in_dataset
            )

            # Step 3: Sample utterance uniformly from
            # the selected dataset-category combination
            if (
                category in self.dataset_category_utts[dataset]
                and self.dataset_category_utts[dataset][category]
            ):
                utts_list = self.dataset_category_utts[dataset][category]
                idx = dataset_category_ptrs[dataset][category] % len(utts_list)
                utt = utts_list[idx]
                dataset_category_ptrs[dataset][category] += 1
                sampled_utts.append(utt)

        # Patch sampled utterances into batches
        self.batch_list = []
        current_batch = []
        current_batch_bins = 0

        for utt in sampled_utts:
            if utt not in utt2sizes[0]:
                continue

            utt_size = utt2sizes[0][utt][0]

            current_batch.append(utt)
            current_batch_bins += utt_size

            if (
                current_batch_bins > self.batch_bins
                and len(current_batch) >= self.min_batch_size
            ) or (
                self.max_batch_size is not None
                and len(current_batch) >= self.max_batch_size
            ):
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_bins = 0

        # Handle last batch
        if not self.drop_last and len(current_batch) >= self.min_batch_size:
            self.batch_list.append(current_batch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"min_batch_size={self.min_batch_size}, "
            f"max_batch_size={self.max_batch_size}, "
            f"category_upsampling_factor={self.category_upsampling_factor}, "
            f"dataset_upsampling_factor={self.dataset_upsampling_factor}), "
            f"dataset_scaling_factor={self.dataset_scaling_factor}, "
            f"drop_last={self.drop_last}, "
            f"category2utt_file={self.category2utt_file}, "
            f"dataset2utt_file={self.dataset2utt_file}"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
