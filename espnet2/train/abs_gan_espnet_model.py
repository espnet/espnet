# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ESPnetModel abstract class for GAN-based training."""

from abc import ABC, abstractmethod
from typing import Dict, Union

import torch

from espnet2.train.abs_espnet_model import AbsESPnetModel


class AbsGANESPnetModel(AbsESPnetModel, torch.nn.Module, ABC):
    """The common abstract class among each GAN-based task.

    "ESPnetModel" is referred to a class which inherits torch.nn.Module,
    and makes the dnn-models "forward" as its member field, a.k.a delegate
    pattern. And "forward" must accept the argument "forward_generator" and
    Return the dict of "loss", "stats", "weight", and "optim_idx".
    "optim_idx" for generator must be 0 and that for discriminator must be 1.

    Example:
        >>> from espnet2.tasks.abs_task import AbsTask
        >>> class YourESPnetModel(AbsGANESPnetModel):
        ...     def forward(self, input, input_lengths, forward_generator=True):
        ...         ...
        ...         if forward_generator:
        ...             # return loss for the generator
        ...             # optim idx 0 indicates generator optimizer
        ...             return dict(loss=loss, stats=stats, weight=weight, optim_idx=0)
        ...         else:
        ...             # return loss for the discriminator
        ...             # optim idx 1 indicates discriminator optimizer
        ...             return dict(loss=loss, stats=stats, weight=weight, optim_idx=1)
        >>> class YourTask(AbsTask):
        ...     @classmethod
        ...     def build_model(cls, args: argparse.Namespace) -> YourESPnetModel:
    """

    @abstractmethod
    def forward(
        self, forward_generator: bool = True, **batch: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """Return the generator loss or the discrimiantor loss.

        This method must have an argument "forward_generator" to switch the generator
        loss calculation and the discrimiantor loss calculation. If forward_generator
        is true, return the generator loss with optim_idx 0. If forward_generator is
        false, return the discrimiantor loss with optim_idx 1.

        Args:
            forward_generator (bool): Whether to return the generator loss or the
                discrimiantor loss. This must have the default value.

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        raise NotImplementedError

    @abstractmethod
    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
