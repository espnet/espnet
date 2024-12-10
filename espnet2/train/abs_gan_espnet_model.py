# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ESPnetModel abstract class for GAN-based training."""

from abc import ABC, abstractmethod
from typing import Dict, Union

import torch

from espnet2.train.abs_espnet_model import AbsESPnetModel


class AbsGANESPnetModel(AbsESPnetModel, torch.nn.Module, ABC):
    """
        ESPnetModel abstract class for GAN-based training.

    This class serves as a common abstract base class for all GAN-based tasks
    within the ESPnet framework. It inherits from both `AbsESPnetModel` and
    `torch.nn.Module`. The `forward` method must accept the argument
    `forward_generator`, which indicates whether to compute the generator
    loss or the discriminator loss. The method should return a dictionary
    containing the "loss", "stats", "weight", and "optim_idx". The
    `optim_idx` should be 0 for the generator and 1 for the discriminator.

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
        ...

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Raises:
        NotImplementedError: If the method is not implemented in a subclass.

    Note:
        This class is intended for use as a base class for creating GAN-based
        models and should not be instantiated directly.

    Todo:
        Implement the forward method in subclasses to handle specific GAN tasks.
    """

    @abstractmethod
    def forward(
        self,
        forward_generator: bool = True,
        **batch: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """
            Return the generator loss or the discriminator loss.

        This method must have an argument "forward_generator" to switch the generator
        loss calculation and the discriminator loss calculation. If forward_generator
        is true, return the generator loss with optim_idx 0. If forward_generator is
        false, return the discriminator loss with optim_idx 1.

        Args:
            forward_generator (bool): Whether to return the generator loss or the
                discriminator loss. This must have the default value.

        Returns:
            Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        Examples:
            >>> model = YourESPnetModel()
            >>> generator_loss = model.forward(input_data, input_lengths, True)
            >>> discriminator_loss = model.forward(input_data, input_lengths, False)

        Note:
            This method is intended to be overridden in subclasses implementing
            specific GAN-based tasks.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
            Collect features from the provided batch of data.

        This method is designed to extract relevant features from the input data
        during the GAN training process. The specific implementation will depend
        on the derived class that overrides this method.

        Args:
            **batch (torch.Tensor): A variable number of keyword arguments
                representing the input data batch. The specific structure
                and contents of the batch depend on the derived class.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the extracted
                features. The keys and values will depend on the specific
                implementation in the derived class.

        Examples:
            >>> class YourESPnetModel(AbsGANESPnetModel):
            ...     def collect_feats(self, **batch):
            ...         # Assume the batch contains a tensor 'x'
            ...         features = some_feature_extraction_method(batch['x'])
            ...         return {'features': features}

            >>> model = YourESPnetModel()
            >>> batch = {'x': torch.randn(10, 3, 64, 64)}  # Example input batch
            >>> features = model.collect_feats(**batch)
            >>> print(features['features'].shape)  # Should print the shape of features
        """
        raise NotImplementedError
