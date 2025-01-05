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
    """
        Round down a number to the nearest multiple of a specified divisor.

    This function takes a number and reduces it to the nearest lower multiple of the
    given divisor. This is particularly useful in scenarios where you need to ensure
    that a quantity is evenly divisible by another.

    Args:
        num (int or float): The number to be rounded down.
        divisor (int or float): The divisor to which the number should be rounded down.

    Returns:
        int or float: The largest integer or float less than or equal to `num` that
        is divisible by `divisor`.

    Examples:
        >>> round_down(10, 3)
        9
        >>> round_down(10.5, 2)
        10.0
        >>> round_down(15, 5)
        15

    Note:
        If the `divisor` is zero, this function will raise a ZeroDivisionError.
    """
    return num - (num % divisor)


class CategoryBalancedSampler(AbsSampler):
    """
        Sampler that maintains an equal distribution of categories (i.e., classes)
    within each minibatch. If the batch size is smaller than the number of
    classes, all samples in the minibatch will belong to different classes.

    The `key_file` is a text file that describes each sample name. It should be
    formatted as follows:

        utterance_id_a
        utterance_id_b
        utterance_id_c

    The first column is referred to, so a 'shape file' can also be used, which
    has the following format:

        utterance_id_a 100,80
        utterance_id_b 400,80
        utterance_id_c 512,80

    Attributes:
        batch_size (int): The size of each batch.
        min_batch_size (int): The minimum size of each batch. Default is 1.
        drop_last (bool): Whether to drop the last batch if it's smaller than
            batch_size. Default is False.
        category2utt_file (Optional[str]): Path to the file mapping categories
            to utterances.
        epoch (int): The epoch number for seeding the random number generator.

    Args:
        batch_size (int): The size of the minibatch.
        min_batch_size (int, optional): Minimum size of the minibatch. Default is 1.
        drop_last (bool, optional): If True, drop the last batch if it's smaller
            than batch_size. Default is False.
        category2utt_file (str, optional): Path to the file mapping categories
            to utterances. Must be provided.
        epoch (int, optional): The epoch number for random seed. Default is 1.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Iterator[Tuple[str, ...]]: An iterator that yields batches of utterances.

    Examples:
        >>> sampler = CategoryBalancedSampler(batch_size=4, category2utt_file='path/to/file.txt')
        >>> for batch in sampler:
        ...     print(batch)

    Note:
        The random seed is initialized based on the provided epoch number to ensure
        reproducibility across different runs.

    Todo:
        - Add support for handling class imbalance.
    """

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
