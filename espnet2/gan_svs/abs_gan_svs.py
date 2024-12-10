# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based SVS abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Union

import torch

from espnet2.svs.abs_svs import AbsSVS


class AbsGANSVS(AbsSVS, ABC):
    """
    Abstract class for GAN-based Singing Voice Synthesis (SVS) models.

    This class serves as a base for implementing GAN-based models for singing
    voice synthesis. It inherits from the `AbsSVS` class and enforces the
    implementation of the `forward` method, which computes the loss for
    either the generator or the discriminator in the GAN framework.

    Attributes:
        None

    Args:
        forward_generator: A callable that represents the generator function.
        *args: Additional positional arguments to be passed to the generator.
        **kwargs: Additional keyword arguments to be passed to the generator.

    Returns:
        A dictionary containing:
            - A tensor representing the generator or discriminator loss.
            - Optionally, a dictionary of tensors related to losses.
            - An integer that may represent the status or additional information.

    Yields:
        None

    Raises:
        NotImplementedError: If the forward method is not implemented in a
        subclass.

    Examples:
        class MyGANSVS(AbsGANSVS):
            def forward(self, forward_generator, *args, **kwargs):
                # Implementation of the forward method
                pass

    Note:
        This class is intended to be subclassed and should not be instantiated
        directly.

    Todo:
        - Implement specific GAN architectures in subclasses.
    """

    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """
        Compute the generator or discriminator loss based on the given inputs.

        This method must be implemented by subclasses to define the specific
        behavior of the forward pass in the GAN-based SVS model. The method
        takes a generator function and additional arguments to calculate
        the losses.

        Args:
            forward_generator (Callable): A function that defines the forward
                pass of the generator or discriminator.
            *args: Variable length argument list to be passed to the
                forward_generator.
            **kwargs: Arbitrary keyword arguments to be passed to the
                forward_generator.

        Returns:
            Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
                A dictionary containing the loss values. The structure of the
                dictionary may vary depending on the implementation, but it
                should include keys for the generator and discriminator losses,
                as well as any additional metrics or values.

        Raises:
            NotImplementedError: If the method is not implemented by the
                subclass.

        Examples:
            To use this method, a subclass must implement it as follows:

            ```python
            class MyGANSVS(AbsGANSVS):
                def forward(self, forward_generator, *args, **kwargs):
                    # Implement the forward logic
                    loss = forward_generator(*args, **kwargs)
                    return {'loss': loss}
            ```

        Note:
            The actual computation of the losses will depend on the specific
            implementation of the `forward_generator`.
        """
        raise NotImplementedError
