import logging
from typing import Callable, Collection, Iterator

import numpy as np
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory


class MultipleIterFactory(AbsIterFactory):
    def __init__(
        self,
        build_funcs: Collection[Callable[[], AbsIterFactory]],
        seed: int = 0,
        shuffle: bool = False,
        validate_each_iter_factory: bool = False,
    ):
        assert check_argument_types()
        self.build_funcs = list(build_funcs)
        self.seed = seed
        self.shuffle = shuffle
        self.completed_iters = []
        self.validate_each_iter_factory = validate_each_iter_factory

    def build_iter(self, epoch: int, shuffle: bool = None) -> Iterator:
        if shuffle is None:
            shuffle = self.shuffle

        build_funcs = list(self.build_funcs)

        if shuffle:
            np.random.RandomState(epoch + self.seed).shuffle(build_funcs)

        for i, build_func in enumerate(build_funcs):

            # Train through all dataset shards and then validate
            if not self.validate_each_iter_factory:
                logging.info(f"Building {i}th iter-factory...")
                iter_factory = build_func()
                assert isinstance(iter_factory, AbsIterFactory), type(iter_factory)
                yield from iter_factory.build_iter(epoch, shuffle)

            # Validate after training through each dataset shard
            else:
                # iterated through all splits, reset again
                if len(self.completed_iters) == len(build_funcs):
                    self.completed_iters = []

                # skip this split if we have unseen splits
                if i in self.completed_iters:
                    continue
                else:
                    logging.info(f"Building {i}th iter-factory...")
                    iter_factory = build_func()
                    assert isinstance(iter_factory, AbsIterFactory), type(iter_factory)
                    self.completed_iters.append(i)
                    yield from iter_factory.build_iter(epoch, shuffle)
                    break
