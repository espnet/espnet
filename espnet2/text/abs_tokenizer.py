from abc import ABC, abstractmethod
from typing import Iterable, List

import torch


class AbsTokenizer(ABC, torch.nn.Module):
    @abstractmethod
    def text2tokens(self, line: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError
