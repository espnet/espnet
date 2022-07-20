from abc import ABC, abstractmethod
from typing import Iterator, Tuple

from torch.utils.data import Sampler


class AbsSampler(Sampler, ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        raise NotImplementedError

    def generate(self, seed):
        return list(self)
