from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsDecoder(torch.nn.Module, ABC):
    """
    Abstract base class for decoders in the ESPnet ASVSpoof framework.

    This class serves as a blueprint for all decoder implementations in the
    ESPnet ASVSpoof project. It inherits from PyTorch's nn.Module and
    requires subclasses to implement the `forward` method, which defines
    the computation performed at every call.

    Attributes:
        None

    Args:
        input (torch.Tensor): The input tensor representing the features to be
            decoded. Its shape should be compatible with the decoder's
            requirements.
        ilens (torch.Tensor): A tensor containing the lengths of the input
            sequences. This is used to manage variable-length inputs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - The first tensor represents the output of the decoder.
            - The second tensor represents the output lengths.

    Yields:
        None

    Raises:
        NotImplementedError: If the `forward` method is called directly
            on the abstract class without being overridden by a subclass.

    Examples:
        Here is an example of how to implement a subclass of AbsDecoder:

        ```python
        class MyDecoder(AbsDecoder):
            def forward(self, input: torch.Tensor, ilens: torch.Tensor) ->
                Tuple[torch.Tensor, torch.Tensor]:
                # Custom decoding logic goes here
                output = ...  # process input
                output_lengths = ...  # compute output lengths
                return output, output_lengths
        ```

    Note:
        Subclasses must provide an implementation of the `forward` method
        to be instantiated.

    Todo:
        Consider adding more specific methods or properties to the
        subclasses to enhance functionality.
    """

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the decoder model.

        This method takes an input tensor and its corresponding lengths, processes
        the input through the decoder, and returns the output tensor along with
        the updated lengths.

        Args:
            input (torch.Tensor): A tensor containing the input data to the decoder.
                The shape should be (batch_size, sequence_length, feature_dimension).
            ilens (torch.Tensor): A tensor containing the lengths of each input
                sequence in the batch. The shape should be (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The output tensor from the decoder, with
                shape (batch_size, output_sequence_length, output_dimension).
                - olens (torch.Tensor): A tensor containing the lengths of each
                output sequence in the batch. The shape will be (batch_size,).

        Raises:
            NotImplementedError: If this method is called directly on the
            AbsDecoder class, as it is intended to be overridden in derived
            classes.

        Examples:
            >>> decoder = MyDecoder()  # MyDecoder should inherit from AbsDecoder
            >>> input_tensor = torch.randn(32, 10, 256)  # Example input
            >>> input_lengths = torch.tensor([10] * 32)  # All sequences are of length 10
            >>> output, output_lengths = decoder(input_tensor, input_lengths)
            >>> print(output.shape)  # Output shape may vary depending on implementation

        Note:
            This method must be implemented in subclasses of AbsDecoder to define
            the specific behavior of the decoder.
        """
        raise NotImplementedError
