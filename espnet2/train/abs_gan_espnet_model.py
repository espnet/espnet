# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based ESPnetModel abstract class."""

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Union

import torch

from espnet2.train.abs_espnet_model import AbsESPnetModel


class AbsGANESPnetModel(AbsESPnetModel, torch.nn.Module, ABC):
    """The common abstract class among each GAN-based task.

    "ESPnetModel" is referred to a class which inherits torch.nn.Module,
    and makes the dnn-models "forward" with "forward_generator" arguments
    as its member field, a.k.a delegate pattern, and defines "loss", "stats",
    "weight", and "optim_idx" for the task. "optim_idx" for generator must be
    0 and that for discriminator must be 1.

    Example:
        >>> from espnet2.tasks.abs_task import AbsTask
        >>> class YourESPnetModel(AbsGANESPnetModel):
        ...     def forward(self, input, input_lengths, forward_generator=True):
        ...         ...
        ...         if forward_generator:
        ...             # optim idx 0 indicates generator optimizer
        ...             return dict(loss=loss, stats=stats, weight=weight, optim_idx=0)
        ...         else:
        ...             # optim idx 1 indicates discriminator optimizer
        ...             return dict(loss=loss, stats=stats, weight=weight, optim_idx=1)
        >>> class YourTask(AbsTask):
        ...     @classmethod
        ...     def build_model(cls, args: argparse.Namespace) -> YourESPnetModel:
    """

    @abstractmethod
    def forward(
        self,
        forward_generator: bool,
        **batch: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        raise NotImplementedError

    @abstractmethod
    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
