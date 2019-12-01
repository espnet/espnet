from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch


class AbsModelController(torch.nn.Module, ABC):
    """The common abstract class among each tasks

    FIXME(kamo): Is Controller a good name?

    "Controller" is referred to as a class which inherits torch.nn.Module,
    and makes the dnn-models forward as its member field,
    a.k.a delegate pattern,
    and defines "loss", "stats", and "weight" for the task.

    If you intend to implement new task in ESPNet,
    the model must inherit this class.
    In other words, the "mediator" objects between
    our training system and the your task class are
    just only these three values.

    Example:
        >>> from espnet2.tasks.abs_task import AbsTask
        >>> class YourController(AbsModelController):
        ...     def forward(self, input, input_lengths):
        ...         ...
        ...         return loss, stats, weight
        >>> class YourTask(AbsTask):
        ...     @classmethod
        ...     def build_model(cls, args: argparse.Namespace) \
        ...             -> YourController:
    """

    @abstractmethod
    def forward(self, **batch: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError
