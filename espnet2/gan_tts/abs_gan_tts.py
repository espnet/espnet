# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based TTS abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Union

import torch

from espnet2.tts.abs_tts import AbsTTS


class AbsGANTTS(AbsTTS, ABC):
    """
        Abstract class for GAN-based Text-to-Speech (TTS) models.

    This class serves as a blueprint for implementing GAN-based TTS models.
    It inherits from the `AbsTTS` class and requires the implementation of
    the `forward` method, which is responsible for calculating the loss
    for either the generator or the discriminator.

    Attributes:
        None

    Args:
        forward_generator: A callable that generates the output from the
            TTS model.
        *args: Additional positional arguments to be passed to the
            forward generator.
        **kwargs: Additional keyword arguments to be passed to the
            forward generator.

    Returns:
        A dictionary containing either:
            - A tensor representing the generator loss.
            - A dictionary of tensors for various losses.
            - An integer indicating the current epoch or step.

    Yields:
        None

    Raises:
        NotImplementedError: If the `forward` method is not implemented
            in the subclass.

    Examples:
        >>> class MyGANTTS(AbsGANTTS):
        ...     def forward(self, forward_generator, *args, **kwargs):
        ...         # Implement the forward logic here
        ...         return {"loss": torch.tensor(0.0)}

        >>> gan_tts = MyGANTTS()
        >>> loss = gan_tts.forward(my_forward_generator, arg1, arg2)
        >>> print(loss)

    Note:
        Subclasses must implement the `forward` method to define the
        specific behavior of the GAN-based TTS model.

    Todo:
        - Implement additional utility methods for handling TTS-specific
          tasks in subclasses.
    """

    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """
                Returns the generator or discriminator loss for the GAN-based TTS model.

        This method is an abstract method that must be implemented by any subclass
        of `AbsGANTTS`. The implementation should define how to compute the loss
        based on the generator's output and any additional inputs provided.

        Args:
            forward_generator: The generator function that produces output
                from the input data.
            *args: Variable length argument list for additional inputs
                required by the generator.
            **kwargs: Arbitrary keyword arguments that may be needed for
                the generator function.

        Returns:
            A dictionary containing:
                - A torch.Tensor representing the loss value.
                - A nested dictionary of torch.Tensor(s) if multiple losses
                  are computed.
                - An integer representing any additional metric, if applicable.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            Here is an example of how to use the forward method in a subclass:

            ```python
            class MyGANTTS(AbsGANTTS):
                def forward(self, forward_generator, *args, **kwargs):
                    # Call the generator and compute loss
                    output = forward_generator(*args, **kwargs)
                    loss = self.compute_loss(output)
                    return {'loss': loss}
            ```

        Note:
            Subclasses must provide an implementation of this method to
            function correctly as a GAN-based TTS model.
        """
        raise NotImplementedError
