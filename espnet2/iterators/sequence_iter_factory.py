from typing import Any
from typing import Sequence

import numpy as np
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory


class SequenceIterFactory(AbsIterFactory):
    """Build iterator for each epoch.

    This class simply creates pytorch DataLoader except for the following points:
    - The random seed is decided according to the number of epochs. This feature
      guarantees reproducibility when resuming from middle of training process.
    - Enable to restrict the number of samples for one epoch. This features
      controls the interval number between training and evaluation.

    """

    def __init__(
        self,
        dataset,
        batches: Sequence[Sequence[Any]],
        num_iters_per_epoch: int = None,
        seed: int = 0,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
    ):
        assert check_argument_types()

        self.batches = list(batches)
        self.dataset = dataset
        if num_iters_per_epoch is not None and num_iters_per_epoch < len(batches):
            self.num_iters_per_epoch = num_iters_per_epoch
        else:
            self.num_iters_per_epoch = None
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
        self.pin_memory = pin_memory

    def build_iter(self, epoch: int, shuffle: bool = None) -> DataLoader:
        if shuffle is None:
            shuffle = self.shuffle

        if self.num_iters_per_epoch is not None:
            N = len(self.batches)
            real_epoch, offset = divmod(self.num_iters_per_epoch * epoch, N)

            if offset >= self.num_iters_per_epoch:
                current_batches = list(self.batches)
                if shuffle:
                    np.random.RandomState(real_epoch + self.seed).shuffle(
                        current_batches
                    )
                batches = current_batches[offset - self.num_iters_per_epoch : offset]
            else:
                prev_batches = list(self.batches)
                current_batches = list(self.batches)
                if shuffle:
                    np.random.RandomState(real_epoch - 1 + self.seed).shuffle(
                        prev_batches
                    )
                    np.random.RandomState(real_epoch + self.seed).shuffle(
                        current_batches
                    )
                batches = (
                    prev_batches[offset - self.num_iters_per_epoch :]
                    + current_batches[:offset]
                )
        else:
            batches = list(self.batches)
            if shuffle:
                np.random.RandomState(epoch + self.seed).shuffle(batches)

        # For backward compatibility for pytorch DataLoader
        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        return DataLoader(
            dataset=self.dataset,
            batch_sampler=batches,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **kwargs,
        )
