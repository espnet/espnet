import math
import numpy as np
from typing import Iterator, List, Optional, Sequence, Tuple, Union
from typeguard import typechecked

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class WeightedBatchSampler(AbsSampler):
    @typechecked
    def __init__(
        self,
        batch_size: int,
        utt2weight_file: str,
    ):
        assert batch_size > 0

        self.batch_size = batch_size
        self.np_seed = 0

        # utt2shape: Weight
        #    uttA 10.0
        #    uttB 20.1
        utt2weight = load_num_sequence_text(utt2weight_file, loader_type="text_float")

        self.keys = sorted(utt2weight, key=lambda k: k)  # sort by keyname
        if len(self.keys) == 0:
            raise RuntimeError(f"0 lines found: {utt2weight_file}")

        self.weights = [utt2weight[k][0] for k in self.keys]
        self.weights = np.array(self.weights) / np.sum(self.weights)
        self.n_batches = math.ceil(len(self.keys) / self.batch_size)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={self.n_batches}, "
            f"batch_size={self.batch_size}, "
        )

    def generate_batch_list_(self) -> List[Tuple[str, ...]]:
        # Sample instances based on weights
        np.random.seed(self.np_seed)
        self.np_seed += 1
        indices = np.random.choice(len(self.keys), len(self.keys), p=self.weights)
        batch_list = []
        cur_batch = []
        for idx in indices:
            cur_batch.append(self.keys[idx])
            if len(cur_batch) == self.batch_size:
                batch_list.append(tuple(cur_batch))
                cur_batch = []
        if len(cur_batch) > 0:
            batch_list.append(tuple(cur_batch))
        assert len(batch_list) == self.n_batches
        return batch_list

    def __len__(self):
        return self.n_batches

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        batch_list = self.generate_batch_list_()
        return iter(batch_list)
