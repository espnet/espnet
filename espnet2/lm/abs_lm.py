from abc import abstractmethod, ABC
from typing import Tuple

import torch

from espnet.nets.scorer_interface import ScorerInterface


class AbsLM(torch.nn.Module, ScorerInterface, ABC):
    """The abstract LM class

    To share the loss calculation way among different models,
    We uses delegate pattern:
    The instance of this class should be passed to "LanguageModel"
    (This naming is confusing, sorry.)

    >>> from espnet2.lm.model import LanguageModel
    >>> lm = AbsLM()
    >>> model = LanguageModel(lm=lm)

    This "model" is a mediator objects for "Task" class.

    """
    @abstractmethod
    def forward(self, input: torch.Tensor, hidden: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
