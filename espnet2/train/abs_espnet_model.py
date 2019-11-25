from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch


class AbsESPNetModel(torch.nn.Module, ABC):
    """The common abstract class among each tasks

    "Model" is referred to as a class which inherits
    torch.nn.Module and define "loss", "stats", and "weight"
    for the task of the model.

    If you intend to implement new task in ESPNet,
    the model must inherit this class.
    In other words, the "mediator" objects between
    our training system and the your task class are
    just only these three values.

    Example:
        >>> from espnet2.tasks.base_task import BaseTask
        >>> class YourTask(BaseTask): pass
        ...     def forward(self, input, input_lengths):
        ...         ...
        ...         return loss, stats, weight
        >>> class YourModel(AbsESPNetModel): pass
        ...     def forward(self, input, input_lengths):
        ...         ...
        ...         return loss, stats, weight


    """

    @abstractmethod
    def forward(self, **kwargs: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError
        return loss, stats, weight
