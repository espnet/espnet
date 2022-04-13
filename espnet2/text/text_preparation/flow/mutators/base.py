from typing import Set
import abc


class AbstractMutator(abc.ABC):
    dependencies = set()

    @abc.abstractmethod
    def __call__(self, text: str) -> str:
        raise NotImplementedError("the `__call__` method must be reloaded")

    @classmethod
    def get_dependencies(cls) -> Set:
        return cls.dependencies
