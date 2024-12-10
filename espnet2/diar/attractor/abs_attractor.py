from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsAttractor(torch.nn.Module, ABC):
    """
    Abstract base class for implementing attractor mechanisms in a neural network.

    This class serves as a blueprint for creating specific types of attractors
    used in speech processing tasks, particularly in the context of speaker
    diarization. It inherits from `torch.nn.Module` and requires subclasses
    to implement the `forward` method.

    Attributes:
        None

    Args:
        enc_input (torch.Tensor): The encoded input tensor, typically containing
            features from a speech signal.
        ilens (torch.Tensor): A tensor containing the lengths of the input
            sequences.
        dec_input (torch.Tensor): The input tensor for the decoder, which may
            include previous decoder outputs or additional context.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output (torch.Tensor): The output tensor from the attractor.
            - attention_weights (torch.Tensor): The attention weights
            corresponding to the attractor's output.

    Raises:
        NotImplementedError: If the `forward` method is not implemented in a
        subclass.

    Examples:
        To create a specific attractor, you would subclass `AbsAttractor` and
        implement the `forward` method as follows:

        ```python
        class MyAttractor(AbsAttractor):
            def forward(self, enc_input, ilens, dec_input):
                # Implement the forward logic here
                return output, attention_weights
        ```

    Note:
        This class is intended to be subclassed. Direct instantiation of
        `AbsAttractor` will result in an error due to the unimplemented
        `forward` method.

    Todo:
        - Implement additional methods or properties that may be needed for
        specific attractor implementations.
    """

    @abstractmethod
    def forward(
        self,
        enc_input: torch.Tensor,
        ilens: torch.Tensor,
        dec_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Computes the forward pass for the AbsAttractor model.

        This method is an abstract method that must be implemented by any subclass
        of the AbsAttractor class. It processes the input tensors to produce the
        output tensors necessary for the model's operation. The specifics of
        the computation will depend on the implementation in the derived class.

        Args:
            enc_input (torch.Tensor): The encoded input tensor, typically from an
                encoder network.
            ilens (torch.Tensor): A tensor representing the lengths of the input
                sequences. This is used to handle variable-length sequences.
            dec_input (torch.Tensor): The input tensor for the decoder, usually
                containing the previous output tokens in the decoding process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - The first tensor is the output of the model after processing the
                input.
                - The second tensor may represent additional information, such as
                hidden states or attention weights.

        Raises:
            NotImplementedError: This method must be implemented by subclasses
                of AbsAttractor.

        Examples:
            class MyAttractor(AbsAttractor):
                def forward(self, enc_input, ilens, dec_input):
                    # Implement specific forward pass logic here
                    return output, additional_info

            model = MyAttractor()
            output, info = model(enc_input, ilens, dec_input)

        Note:
            This method is intended to be overridden in derived classes to
            provide specific functionality for different types of attractors.
        """
        raise NotImplementedError
