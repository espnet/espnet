import itertools
import random
from functools import partial
from typing import Any, Optional, Sequence, Union

import numpy as np
from torch.utils.data import DataLoader
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.samplers.abs_sampler import AbsSampler


def worker_init_fn(worker_id, base_seed=0):
    """
    Set random seed for each worker in DataLoader.

    This function initializes the random seed for each worker process used by
    PyTorch's DataLoader. It ensures that each worker has a unique seed, which
    is calculated based on the base seed and the worker ID. This is particularly
    useful for ensuring reproducibility in data loading across different runs
    of the training process.

    Args:
        worker_id (int): The ID of the worker. Each worker will have a
            unique ID, starting from 0.
        base_seed (int, optional): The base seed used for generating the
            random seed for each worker. Default is 0.

    Examples:
        >>> worker_init_fn(0, base_seed=42)  # Initializes the seed for worker 0
        >>> worker_init_fn(1, base_seed=42)  # Initializes the seed for worker 1

    Note:
        This function is typically passed to the `DataLoader` as the
        `worker_init_fn` argument.
    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


class RawSampler(AbsSampler):
    """
        RawSampler is a class that inherits from AbsSampler and provides a mechanism to
    sample raw batches of data.

    Attributes:
        batches (Sequence[Sequence[Any]]): A sequence of batches to be sampled.

    Args:
        batches (Sequence[Sequence[Any]]): A sequence of batches that the sampler
            will iterate over.

    Returns:
        None

    Yields:
        Sequence[Any]: The next batch of data during iteration.

    Raises:
        None

    Examples:
        >>> sampler = RawSampler([[1, 2], [3, 4], [5, 6]])
        >>> len(sampler)
        3
        >>> list(sampler)
        [[1, 2], [3, 4], [5, 6]]
        >>> sampler.generate(seed=42)
        [[1, 2], [3, 4], [5, 6]]

    Note:
        The `generate` method returns the list of batches without any
        modifications based on the seed.
    """

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def generate(self, seed):
        """
            Generate a list of batches based on the provided seed.

        This method returns the batches that were initialized in the
        RawSampler. The seed can be used to ensure that the same
        batches are produced consistently, allowing for reproducibility
        in experiments.

        Args:
            seed (int): The seed used for generating the batches. This
                seed is added to the base seed to ensure variability in
                the generated batches across different epochs.

        Returns:
            list: A list containing the batches as defined in the
                RawSampler.

        Examples:
            >>> sampler = RawSampler([[1, 2], [3, 4], [5, 6]])
            >>> batches = sampler.generate(seed=42)
            >>> print(batches)
            [[1, 2], [3, 4], [5, 6]]
        """
        return list(self.batches)


class SequenceIterFactory(AbsIterFactory):
    """
    Build iterator for each epoch.

    This class creates a PyTorch DataLoader with additional features for
    reproducibility and control over the number of samples per epoch. The
    following points are noteworthy:

    - The random seed is determined based on the epoch number, ensuring
      reproducibility when resuming training.
    - It allows restriction on the number of samples for each epoch,
      facilitating control over the interval between training and evaluation.

    Attributes:
        dataset (Any): The dataset to be used for the DataLoader.
        sampler (AbsSampler): An instance of a sampler that defines the order
            in which data is drawn.
        num_iters_per_epoch (Optional[int]): The number of iterations to
            perform in one epoch.
        seed (int): The seed for random number generation.
        shuffle (bool): Whether to shuffle the data at the beginning of each
            epoch.
        shuffle_within_batch (bool): Whether to shuffle the data within each
            batch.
        num_workers (int): The number of worker processes for data loading.
        collate_fn (Optional[callable]): Function to merge a list of samples
            into a batch.
        pin_memory (bool): If True, the data loader will copy Tensors into
            pinned memory before returning them.

    Args:
        dataset (Any): The dataset from which to load the data.
        batches (Union[AbsSampler, Sequence[Sequence[Any]]]): The batches of
            data to be sampled.
        num_iters_per_epoch (Optional[int]): The number of iterations per
            epoch (default is None).
        seed (int): Random seed (default is 0).
        shuffle (bool): Whether to shuffle the data (default is False).
        shuffle_within_batch (bool): Whether to shuffle data within each
            batch (default is False).
        num_workers (int): Number of subprocesses to use for data loading
            (default is 0).
        collate_fn (Optional[callable]): Function to merge a list of samples
            into a batch (default is None).
        pin_memory (bool): If True, the data loader will copy Tensors into
            pinned memory (default is False).

    Returns:
        DataLoader: A PyTorch DataLoader instance configured with the
        specified parameters.

    Examples:
        >>> factory = SequenceIterFactory(dataset, batches, num_iters_per_epoch=10)
        >>> data_loader = factory.build_iter(epoch=1)

    Note:
        The `build_iter` method can be called multiple times with different
        epoch values to generate iterators for different training phases.
    """

    @typechecked
    def __init__(
        self,
        dataset,
        batches: Union[AbsSampler, Sequence[Sequence[Any]]],
        num_iters_per_epoch: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = False,
        shuffle_within_batch: bool = False,
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
        self.shuffle = shuffle
        self.shuffle_within_batch = shuffle_within_batch
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
        self.pin_memory = pin_memory

    def build_iter(self, epoch: int, shuffle: bool = None) -> DataLoader:
        """
            Build iterator for each epoch.

        This class simply creates a PyTorch DataLoader with the following features:
        - The random seed is determined based on the epoch number, ensuring
          reproducibility when resuming from a training session.
        - Allows restriction on the number of samples per epoch, controlling
          the interval between training and evaluation.

        Attributes:
            dataset (Any): The dataset to load data from.
            sampler (AbsSampler): The sampler used to sample batches of data.
            num_iters_per_epoch (Optional[int]): Maximum number of iterations per epoch.
            shuffle (bool): Whether to shuffle the data.
            shuffle_within_batch (bool): Whether to shuffle data within each batch.
            seed (int): Seed for random number generation.
            num_workers (int): Number of subprocesses to use for data loading.
            collate_fn (Optional[callable]): Function to merge a list of samples.
            pin_memory (bool): If True, the data loader will copy Tensors into CUDA
                pinned memory.

        Args:
            epoch (int): The current epoch number.
            shuffle (Optional[bool]): If provided, overrides the default shuffle
                setting.

        Returns:
            DataLoader: A PyTorch DataLoader object configured for the current
            epoch.

        Raises:
            AssertionError: If the length of batches does not match
            num_iters_per_epoch when it is specified.

        Examples:
            >>> factory = SequenceIterFactory(dataset=my_dataset, batches=my_batches)
            >>> data_loader = factory.build_iter(epoch=1, shuffle=True)

        Note:
            Ensure that the dataset and batches are compatible with the
            DataLoader's requirements.

        Todo:
            - Add support for custom collate functions.
        """
        if shuffle is None:
            shuffle = self.shuffle

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

        # reshuffle whole 'batches' so that elements within a batch can move
        # between different batches
        if self.shuffle_within_batch:
            _bs = len(batches[0])
            batches = list(itertools.chain(*batches))
            np.random.RandomState(epoch + self.seed).shuffle(batches)
            _batches = []
            for ii in range(0, len(batches), _bs):
                _batches.append(batches[ii : ii + _bs])
            batches = _batches
            del _batches

        return DataLoader(
            dataset=self.dataset,
            batch_sampler=batches,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=partial(worker_init_fn, base_seed=epoch + self.seed),
            **kwargs,
        )
