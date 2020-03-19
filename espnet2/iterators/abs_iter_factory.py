from abc import ABC
from abc import abstractmethod


class AbsIterFactory(ABC):
    @abstractmethod
    def build_iter(self, epoch: int, shuffle: bool = None):
        raise NotImplementedError
