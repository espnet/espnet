# For multi-task learning (MTL) with two tasks, this sampler creates
# mini-batches where half of the samples are from equally distributed
# categories (i.e., classes) for the first task and the other half are
# from equally distributed categories for the second task.
# If the batch_size is smaller than the number of classes,
# all samples in the (half) minibatch will belong to different classes.

import random
from collections import Counter
from typing import Iterator, Optional, Tuple

from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler

def round_down(num, divisor):
    return num - (num % divisor)

class BinaryTaskCategoryBalancedSampler(AbsSampler):
    @typechecked
    def __init__(
        self,
        batch_size: int,
        min_batch_size: int = 1,
        drop_last: bool = False,
        category2utt_file: Optional[str] = None,
        category2utt_file2: Optional[str] = None,
        epoch: int = 1,
        **kwargs,
    ):
        assert batch_size > 0
        assert batch_size % 2 == 0, "Batch size should be an even number."
        random.seed(epoch)

        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.drop_last = drop_last

        assert category2utt_file is not None and category2utt_file2 is not None

        # read the category-to-utterance mappings
        category1_utt = read_2columns_text(category2utt_file)
        category2_utt = read_2columns_text(category2utt_file2)

        # initialize batches for each category
        self.category1_batches = self._create_batches(category1_utt, batch_size // 2)
        self.category2_batches = self._create_batches(category2_utt, batch_size // 2)

        self.batch_list = []
        # combine half batches from both categories to form full batches
        for cat1_batch, cat2_batch in zip(self.category1_batches, self.category2_batches):
            self.batch_list.append(cat1_batch + cat2_batch)

    def _create_batches(self, category_utt, half_batch_size):
        categories = list(category_utt.keys())
        random.shuffle(categories)

        n_utt_per_category_in_batch = max(1, half_batch_size // len(categories))

        flattened_cats = []
        for cat in categories:
            category_utt[cat] = category_utt[cat].split(" ")
            flattened_cats.extend([cat] * len(category_utt[cat]))
            random.shuffle(category_utt[cat])

        rand_idx = list(range(len(flattened_cats)))
        random.shuffle(rand_idx)

        batch_list = []
        current_batch = []
        current_batch_stats = Counter()
        
        # make (half) mini-batches
        for idx in rand_idx:
            # don't allow more number of samples that belong to each category
            # than n_utt_per_category_in_batch
            if current_batch_stats[flattened_cats[idx]] >= n_utt_per_category_in_batch:
                continue

            current_batch.append(category_utt[flattened_cats[idx]].pop())
            current_batch_stats[flattened_cats[idx]] += 1

            if len(current_batch) == half_batch_size:
                batch_list.append(current_batch)
                current_batch = []
                current_batch_stats = Counter()

        return batch_list

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)