import logging
from typing import Callable, Collection, Iterator

import numpy as np
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory


class MultipleIterFactory(AbsIterFactory):
    """
        A factory for creating multiple iterator instances based on provided builder
    functions. This class allows for the optional shuffling of the iterator
    construction order and can be seeded for reproducibility.

    Attributes:
        build_funcs (list): A list of callable functions that return instances of
            AbsIterFactory.
        seed (int): An integer seed for random number generation to control
            shuffling.
        shuffle (bool): A flag indicating whether to shuffle the order of
            iterator factories.

    Args:
        build_funcs (Collection[Callable[[], AbsIterFactory]]): A collection of
            functions that will be called to create iterator factories.
        seed (int, optional): The seed for the random number generator. Defaults
            to 0.
        shuffle (bool, optional): Whether to shuffle the order of iterators.
            Defaults to False.

    Yields:
        Iterator: An iterator that yields items from the constructed iterators.

    Examples:
        >>> def build_func_a():
        ...     return SomeIterFactoryA()
        ...
        >>> def build_func_b():
        ...     return SomeIterFactoryB()
        ...
        >>> factory = MultipleIterFactory([build_func_a, build_func_b], seed=42,
        ...                               shuffle=True)
        >>> for item in factory.build_iter(epoch=1):
        ...     print(item)

    Note:
        The order of the iterators can change between different epochs if
        shuffling is enabled.

    Todo:
        - Add more functionality for better error handling.
        - Implement additional logging options.
    """

    @typechecked
    def __init__(
        self,
        build_funcs: Collection[Callable[[], AbsIterFactory]],
        seed: int = 0,
        shuffle: bool = False,
    ):
        self.build_funcs = list(build_funcs)
        self.seed = seed
        self.shuffle = shuffle

    def build_iter(self, epoch: int, shuffle: bool = None) -> Iterator:
        """
                Generates an iterator by building iterators from a collection of provided
        functions.

        This method allows for the creation of multiple iterators, which can be
        shuffled based on the epoch and a specified seed. The iterator is built by
        calling each of the provided build functions in sequence.

        Args:
            epoch (int): The current epoch number, used to seed the random
                number generator for shuffling.
            shuffle (bool, optional): If True, the build functions will be
                shuffled before building iterators. If None, defaults to the
                instance's shuffle attribute.

        Yields:
            Iterator: An iterator that yields elements from the built iterators.

        Examples:
            >>> factory = MultipleIterFactory([build_func1, build_func2], seed=42)
            >>> for item in factory.build_iter(epoch=1, shuffle=True):
            ...     print(item)

        Note:
            This method is intended for use within the MultipleIterFactory
            context and assumes that the build functions return instances of
            AbsIterFactory.

        Raises:
            AssertionError: If the object returned by a build function is not
                an instance of AbsIterFactory.
        """
        if shuffle is None:
            shuffle = self.shuffle

        build_funcs = list(self.build_funcs)

        if shuffle:
            np.random.RandomState(epoch + self.seed).shuffle(build_funcs)

        for i, build_func in enumerate(build_funcs):
            logging.info(f"Building {i}th iter-factory...")
            iter_factory = build_func()
            assert isinstance(iter_factory, AbsIterFactory), type(iter_factory)
            yield from iter_factory.build_iter(epoch, shuffle)
