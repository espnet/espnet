# Sampler that keeps equally distributed categories (i.e., classes) within
# each minibatch. If the batch_size is smaller than the number of classes,
# all samples in the minibatch will belong to different classes.
# Cross-checked with https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py
from typing import Iterator, List, Sequence, Tuple, Union

from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text, read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler

def round_down(num, divisor):
        return num - (num%divisor)

class CategoryBalancedSampler(AbsSampler):
    def __init__(
        self,
        batch_size: int,
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        category2uttt_file: str = None,
    ):
        assert check_argument_types()
        assert batch_size > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        assert category2utt_file is not None

        # dictionary with categories as keys and corresponding utterances
        # as values
        category2utt = read_2columns_test(category2utt_file)

        categories = list(category2utt.keys())
        assert len(categories) >= self.batch_size, f"batch size is {batch_size} is larger than number of classes {len(categories)}"
        categories.sort()

        while True:
            if len(category2utt) < self.batch_size:
                break






        self.batch_list = []
        for d, v in category2utt.items():
            category_keys = v
            # Decide batch-sizes
            start = 0
            batch_sizes = []
            while True:
                k = category_keys[start]
                factor = max(int(d[k][0] / m) for d, m in zip(utt2shapes, fold_lengths))
                bs = max(min_batch_size, int(batch_size / (1 + factor)))
                if self.drop_last and start + bs > len(category_keys):
                    # This if-block avoids 0-batches
                    if len(self.batch_list) > 0:
                        break

                bs = min(len(category_keys) - start, bs)
                batch_sizes.append(bs)
                start += bs
                if start >= len(category_keys):
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
                assert sum(batch_sizes) == len(
                    category_keys
                ), f"{sum(batch_sizes)} != {len(category_keys)}"

            # Set mini-batch
            cur_batch_list = []
            start = 0
            for bs in batch_sizes:
                assert len(category_keys) >= start + bs, "Bug"
                minibatch_keys = category_keys[start : start + bs]
                start += bs
                if sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending or "
                        f"descending: {sort_in_batch}"
                    )
                cur_batch_list.append(tuple(minibatch_keys))

            if sort_batch == "ascending":
                pass
            elif sort_batch == "descending":
                cur_batch_list.reverse()
            else:
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )
            self.batch_list.extend(cur_batch_list)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
