# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based ESPnetModel abstract class."""

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Union

import torch


class AbsGANESPnetModel(torch.nn.Module, ABC):
    """The common abstract class among each GAN-based task

    "ESPnetModel" is referred to a class which inherits torch.nn.Module,
    and makes the dnn-models "forward_generator" and "forward_discrminator"
    as its member field, a.k.a delegate pattern, and defines "loss", "stats",
    "weight", and "optim_idx" for the task. "optim_idx" for generator must be
    0 and that for discriminator must be 1.

    Example:
        >>> from espnet2.tasks.abs_task import AbsTask
        >>> class YourESPnetModel(AbsGANESPnetModel):
        ...     def forward_generator(self, input, input_lengths):
        ...         ...
        ...         return dict(loss=loss, stats=stats, weight=weight, optim_idx=0)
        ...     def forward_discrminator(self, input, input_lengths):
        ...         ...
        ...         return dict(loss=loss, stats=stats, weight=weight, optim_idx=1)
        >>> class YourTask(AbsTask):
        ...     @classmethod
        ...     def build_model(cls, args: argparse.Namespace) -> YourESPnetModel:
    """

    @abstractmethod
    def forward_generator(
        self, **batch: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        raise NotImplementedError

    @abstractmethod
    def forward_discrminator(
        self, **batch: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        raise NotImplementedError

    @abstractmethod
    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
