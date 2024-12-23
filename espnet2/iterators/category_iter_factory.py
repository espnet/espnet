import random
from functools import partial
from typing import Any, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.samplers.category_balanced_sampler import CategoryBalancedSampler


def worker_init_fn(worker_id, base_seed=0):
    """
    Set random seed for each worker in DataLoader.

    This function initializes the random seed for each worker in the
    PyTorch DataLoader to ensure reproducibility. The seed is calculated
    based on the provided base seed and the worker ID.

    Args:
        worker_id (int): The ID of the worker being initialized.
        base_seed (int, optional): The base seed for random number
            generation. Defaults to 0.

    Examples:
        >>> worker_init_fn(0, base_seed=42)
        >>> worker_init_fn(1, base_seed=42)

    Note:
        This function is typically used in conjunction with the
        `DataLoader` class in PyTorch, especially when using multiple
        workers for data loading.
    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


class RawSampler(AbsSampler):
    """
    Sampler for raw batches of data.

    This class provides a simple implementation of a sampler that yields
    batches directly from the provided input list. It can be used to iterate
    over the batches in a dataset without any additional sampling logic.

    Attributes:
        batches (Sequence[Sequence[Any]]): A sequence of batches to be sampled.

    Args:
        batches (Sequence[Sequence[Any]]): The input batches to be used by the
            sampler.

    Returns:
        int: The number of batches available for sampling.

    Yields:
        Sequence[Any]: The next batch of data when iterated.

    Raises:
        ValueError: If the input batches are empty.

    Examples:
        >>> sampler = RawSampler([[1, 2], [3, 4], [5, 6]])
        >>> list(sampler)
        [[1, 2], [3, 4], [5, 6]]
        >>> len(sampler)
        3
    """

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def generate(self, seed):
        """
            Generate a list of batches with a given random seed.

        This method retrieves the batches of data stored in the sampler. The
        `seed` parameter allows for controlled randomness in the sampling
        process, enabling reproducibility.

        Args:
            seed (int): The random seed used for generating the batches.

        Returns:
            List: A list of batches generated from the sampler.

        Examples:
            >>> sampler = RawSampler([[1, 2], [3, 4], [5, 6]])
            >>> batches = sampler.generate(seed=42)
            >>> print(batches)
            [[1, 2], [3, 4], [5, 6]]

        Note:
            The generated batches will always be the same for the same seed
            and the same underlying data.

        Raises:
            ValueError: If the seed is not a valid integer.
        """
        return list(self.batches)


class CategoryIterFactory(AbsIterFactory):
    """
    Build iterator for each epoch.

    This class creates a PyTorch DataLoader with the following features:
    - The random seed is determined by the epoch number, ensuring reproducibility
      when resuming training.
    - The number of samples for each epoch can be restricted, controlling the
      interval between training and evaluation.

    Attributes:
        dataset (Any): The dataset to be used by the DataLoader.
        sampler (AbsSampler): An instance of a sampler that defines how to sample
            data from the dataset.
        num_iters_per_epoch (int): The number of iterations to run per epoch.
        sampler_args (dict): Arguments to configure the sampler.
        shuffle (bool): Whether to shuffle the data at the beginning of each epoch.
        seed (int): The seed for random number generation.
        num_workers (int): The number of subprocesses to use for data loading.
        collate_fn (callable): Function to merge a list of samples to form a
            mini-batch.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them.

    Args:
        dataset (Any): The dataset from which to load the data.
        batches (Union[AbsSampler, Sequence[Sequence[Any]]]): Either a sampler
            or a sequence of batches to use for sampling.
        num_iters_per_epoch (int, optional): The number of iterations to run
            per epoch. Defaults to None.
        seed (int, optional): The random seed. Defaults to 0.
        sampler_args (dict, optional): Additional arguments for the sampler.
            Defaults to None.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to
            False.
        num_workers (int, optional): Number of worker processes for data loading.
            Defaults to 0.
        collate_fn (callable, optional): Function to merge a list of samples to
            form a mini-batch. Defaults to None.
        pin_memory (bool, optional): If True, data loader will copy Tensors into
            CUDA pinned memory. Defaults to False.

    Returns:
        DataLoader: A PyTorch DataLoader instance configured for the dataset
        and sampler.

    Examples:
        # Create an instance of CategoryIterFactory
        iter_factory = CategoryIterFactory(
            dataset=my_dataset,
            batches=my_batches,
            num_iters_per_epoch=100,
            seed=42,
            shuffle=True
        )

        # Build the DataLoader for epoch 1
        data_loader = iter_factory.build_iter(epoch=1)

    Note:
        This class is designed to work with PyTorch DataLoader and provides
        additional control over the sampling process.

    Raises:
        RuntimeError: If the batch size is less than the world size during
        distributed training.
    """

    @typechecked
    def __init__(
        self,
        dataset,
        batches: Union[AbsSampler, Sequence[Sequence[Any]]],
        num_iters_per_epoch: int = None,
        seed: int = 0,
        sampler_args: dict = None,
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
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
        self.pin_memory = pin_memory

    def build_iter(self, epoch: int, shuffle: bool = None) -> DataLoader:
        """
            Build iterator for each epoch.

        This class simply creates a PyTorch DataLoader with the following features:
        - The random seed is determined according to the number of epochs,
          ensuring reproducibility when resuming training.
        - It allows restriction on the number of samples for one epoch,
          controlling the interval between training and evaluation.

        Attributes:
            dataset (Any): The dataset to be loaded.
            sampler (AbsSampler): The sampler to sample batches from the dataset.
            num_iters_per_epoch (int, optional): The number of iterations per epoch.
            sampler_args (dict, optional): Arguments for the sampler.
            shuffle (bool): Whether to shuffle the data at every epoch.
            seed (int): Random seed for reproducibility.
            num_workers (int): Number of workers for data loading.
            collate_fn (callable, optional): Function to collate data samples.
            pin_memory (bool): Whether to pin memory for faster data transfer.

        Args:
            dataset (Any): The dataset to be used for the iterator.
            batches (Union[AbsSampler, Sequence[Sequence[Any]]]): Batches of data
                or an instance of AbsSampler.
            num_iters_per_epoch (int, optional): Number of iterations per epoch.
            seed (int, optional): Seed for random number generation. Default is 0.
            sampler_args (dict, optional): Additional arguments for the sampler.
            shuffle (bool, optional): Whether to shuffle the dataset. Default is False.
            num_workers (int, optional): Number of worker threads for loading data.
            collate_fn (callable, optional): Function to merge a list of samples
                into a batch.
            pin_memory (bool, optional): If True, the data loader will copy tensors
                into CUDA pinned memory before returning them.

        Returns:
            DataLoader: A PyTorch DataLoader for the specified dataset and batches.

        Examples:
            factory = CategoryIterFactory(dataset=my_dataset, batches=my_batches,
                                          num_iters_per_epoch=100, seed=42)
            dataloader = factory.build_iter(epoch=1, shuffle=True)

        Raises:
            RuntimeError: If the batch size is less than the world size in
                distributed training.
        """
        if shuffle is None:
            shuffle = self.shuffle

        # rebuild sampler
        if epoch > 1:
            self.sampler_args["epoch"] = epoch
            batch_sampler = CategoryBalancedSampler(**self.sampler_args)
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
