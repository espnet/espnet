from typing import Any
from typing import Sequence
from typing import Union
from typing import List
from typing import Iterator

import numpy as np
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.curriculum.curriculum_sampler import CurriculumSampler



class CurriculumIterFactory(AbsIterFactory):
    def __init__(
        self,
        dataset,
        batches: Union[List[Iterator[Any]]],
        num_iters_per_epoch: int = None,
        seed: int = 0,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
    ):

        assert check_argument_types()
       
        self.sampler = batches
        self.dataset = dataset
        self.num_iters_per_epoch = num_iters_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
        self.pin_memory = pin_memory

    
    def build_iter(self, epoch=1):
        #epoch is a dummy variable to accomodate trainer.run() method.
        #Instead of one data loader we return K data loader for each task
        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        loaders = []
        for i in range(len(self.sampler)-1):
            loaders.append(
                DataLoader(
                    dataset=self.dataset,
                    batch_sampler=self.sampler[i],
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    **kwargs,
                )
            )

        return loaders