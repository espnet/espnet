import random
from functools import partial
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.samplers.category_balanced_sampler import CategoryBalancedSampler
from espnet2.samplers.category_power_sampler import (
    CategoryDatasetPowerSampler,
    CategoryPowerSampler,
)


def worker_init_fn(worker_id, base_seed=0):
    """Set random seed for each worker in DataLoader."""
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


class RawSampler(AbsSampler):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def generate(self, seed):
        return list(self.batches)


class CategoryIterFactory(AbsIterFactory):
    """Build iterator for each epoch.

    This class simply creates pytorch DataLoader except for the following points:
    - The random seed is decided according to the number of epochs. This feature
      guarantees reproducibility when resuming from middle of training process.
    - Enable to restrict the number of samples for one epoch. This features
      controls the interval number between training and evaluation.

    Args:
        dataset: The dataset to iterate over
        batches: The batches to iterate over
        num_iters_per_epoch: The number of iterations per epoch
        seed: The random seed
        sampler_args: The arguments to pass to the batch sampler
        batch_type: The type of batch sampler to use:
            catbel: Category-balanced batch sampler,
                    ensures equal representation of all categories in each batch
            catpow: Category-power batch sampler,
                    applies power law sampling based on category frequency to
                    address class imbalance
            catpow_dataset: Category-power batch sampler with dataset-level
                    upsampling, performs dataset-level upsampling before applying
                    power law sampling on categories within each dataset
        shuffle: Whether to shuffle the batches
        num_workers: The number of workers to use
        collate_fn: The collate function to use
        pin_memory: Whether to pin the memory
    """

    @typechecked
    def __init__(
        self,
        dataset,
        batches: Union[AbsSampler, Sequence[Sequence[Any]]],
        num_iters_per_epoch: Optional[int] = None,
        seed: int = 0,
        sampler_args: dict = None,
        batch_type: str = "catbel",  # or "catpow", "catpow_dataset"
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
    ):

        if not isinstance(batches, AbsSampler):
            self.sampler = RawSampler(batches)
        else:
            self.sampler = batches

        self.dataset = dataset
        self.num_iters_per_epoch = num_iters_per_epoch
        self.sampler_args = sampler_args
        self.batch_type = batch_type
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
        self.pin_memory = pin_memory

    def build_iter(self, epoch: int, shuffle: bool = None) -> DataLoader:
        if shuffle is None:
            shuffle = self.shuffle

        # rebuild sampler
        if epoch > 1:
            self.sampler_args["epoch"] = epoch
            if self.batch_type == "catbel":
                batch_sampler = CategoryBalancedSampler(**self.sampler_args)
            elif self.batch_type == "catpow":
                batch_sampler = CategoryPowerSampler(**self.sampler_args)
            elif self.batch_type == "catpow_balance_dataset":
                batch_sampler = CategoryDatasetPowerSampler(**self.sampler_args)
            else:
                raise ValueError(f"Unsupported batch_type: {self.batch_type}")

            batches = list(batch_sampler)

            if self.sampler_args["num_batches"] is not None:
                batches = batches[: self.sampler_args.num_batches]

            if self.sampler_args["distributed"]:
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                for batch in batches:
                    if len(batch) < world_size:
                        raise RuntimeError(
                            f"The batch-size must be equal or more than world_size: "
                            f"{len(batch)} < {world_size}"
                        )
                batches = [batch[rank::world_size] for batch in batches]
            self.sampler = RawSampler(batches)

        if self.num_iters_per_epoch is not None:
            N = len(self.sampler)
            # If corpus size is larger than the num_per_epoch
            if self.num_iters_per_epoch < N:
                N = len(self.sampler)
                real_epoch, offset = divmod(self.num_iters_per_epoch * epoch, N)

                if offset >= self.num_iters_per_epoch:
                    current_batches = self.sampler.generate(real_epoch + self.seed)
                    if shuffle:
                        np.random.RandomState(real_epoch + self.seed).shuffle(
                            current_batches
                        )
                    batches = current_batches[
                        offset - self.num_iters_per_epoch : offset
                    ]
                else:
                    prev_batches = self.sampler.generate(real_epoch - 1 + self.seed)
                    current_batches = self.sampler.generate(real_epoch + self.seed)
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

            # If corpus size is less than the num_per_epoch
            else:
                _epoch, _cursor = divmod(self.num_iters_per_epoch * (epoch - 1), N)
                _remain = self.num_iters_per_epoch
                batches = []
                current_batches = self.sampler.generate(_epoch + self.seed)
                if shuffle:
                    np.random.RandomState(_epoch + self.seed).shuffle(current_batches)
                while _remain > 0:
                    _batches = current_batches[_cursor : _cursor + _remain]
                    batches += _batches
                    if _cursor + _remain >= N:
                        _epoch += 1
                        _cursor = 0
                        current_batches = self.sampler.generate(_epoch + self.seed)
                        if shuffle:
                            np.random.RandomState(_epoch + self.seed).shuffle(
                                current_batches
                            )
                    else:
                        _cursor = _cursor + _remain
                    _remain -= len(_batches)

                assert len(batches) == self.num_iters_per_epoch

        else:
            batches = self.sampler.generate(epoch + self.seed)
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
            worker_init_fn=partial(worker_init_fn, base_seed=epoch + self.seed),
            **kwargs,
        )
