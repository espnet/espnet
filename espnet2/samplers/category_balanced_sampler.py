# Sampler that keeps equally distributed categories (i.e., classes) within
# each minibatch. If the batch_size is smaller than the number of classes,
# all samples in the minibatch will belong to different classes.
# Cross-checked with https://github.com/clovaai/voxceleb_trainer/blob/master/
# DatasetLoader.py
# 'key_file' is just a text file which describes each sample name."
# \n\n"
#     utterance_id_a\n"
#     utterance_id_b\n"
#     utterance_id_c\n"
# \n"
# The fist column is referred, so 'shape file' can be used, too.\n\n"
#     utterance_id_a 100,80\n"
#     utterance_id_b 400,80\n"
#     utterance_id_c 512,80\n",
import random
from collections import Counter
from typing import Iterator, Optional, Tuple

from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler


def round_down(num, divisor):
    return num - (num % divisor)


class CategoryBalancedSampler(AbsSampler):
    @typechecked
    def __init__(
        self,
        batch_size: int,
        min_batch_size: int = 1,
        drop_last: bool = False,
        category2utt_file: Optional[str] = None,
        epoch: int = 1,
        **kwargs,
    ):
        assert batch_size > 0
        random.seed(epoch)

        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.drop_last = drop_last

        assert category2utt_file is not None
        # dictionary with categories as keys and corresponding utterances
        # as values
        category2utt = read_2columns_text(category2utt_file)

        categories = list(category2utt.keys())
        random.shuffle(categories)

        if len(categories) >= self.batch_size:
            n_utt_per_category_in_batch = 1
        else:
            n_utt_per_category_in_batch = int(self.batch_size / len(categories)) + 1

        flattened_cats = []
        for cat in categories:
            category2utt[cat] = category2utt[cat].split(" ")
            flattened_cats.extend([cat] * len(category2utt[cat]))
            random.shuffle(category2utt[cat])

        rand_idx = list(range(len(flattened_cats)))
        random.shuffle(rand_idx)

        self.batch_list = []
        current_batch = []
        current_batch_stats = Counter()
        # make minibatches
        for idx in rand_idx:
            # don't allow more number of samples that belong to each category
            # then n_utt_per_category_in_batch
            if current_batch_stats[flattened_cats[idx]] >= n_utt_per_category_in_batch:
                continue

            current_batch.append(category2utt[flattened_cats[idx]].pop())
            current_batch_stats[flattened_cats[idx]] += 1

            # append batch to batch list
            if len(current_batch) == self.batch_size:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_stats = Counter()

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
