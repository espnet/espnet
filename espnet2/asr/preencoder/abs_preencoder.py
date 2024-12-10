from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsPreEncoder(torch.nn.Module, ABC):
    """
    Abstract base class for pre-encoders in the ESPnet2 ASR framework.

    This class defines the interface for pre-encoder modules that can be used
    within the Automatic Speech Recognition (ASR) pipeline of the ESPnet2
    framework. It inherits from PyTorch's `torch.nn.Module` and enforces the
    implementation of two key methods: `output_size` and `forward`.

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If `output_size` or `forward` methods are not
            implemented in a subclass.

    Methods:
        output_size: Returns the output size of the pre-encoder.
        forward: Processes the input tensor and returns the encoded output.

    Examples:
        To create a custom pre-encoder, subclass `AbsPreEncoder` and implement
        the required methods:

        ```python
        class CustomPreEncoder(AbsPreEncoder):
            def output_size(self) -> int:
                return 128  # Example output size

            def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> 
            Tuple[torch.Tensor, torch.Tensor]:
                # Implement the encoding logic here
                encoded_output = input  # Placeholder for actual encoding
                return encoded_output, input_lengths
        ```

    Note:
        This class is not intended to be instantiated directly. Instead, it
        serves as a blueprint for other pre-encoder implementations.
    """
    @abstractmethod
    def output_size(self) -> int:
        """
        Computes the output size of the pre-encoder.

        This method is intended to be implemented by subclasses of 
        AbsPreEncoder to provide the specific output size after 
        processing the input tensor. The output size is typically 
        determined based on the architecture of the encoder.

        Returns:
            int: The size of the output produced by the pre-encoder.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.

        Examples:
            class CustomPreEncoder(AbsPreEncoder):
                def output_size(self) -> int:
                    return 128  # Example output size

            encoder = CustomPreEncoder()
            print(encoder.output_size())  # Output: 128

        Note:
            This method must be overridden in any concrete subclass of 
            AbsPreEncoder to ensure proper functionality.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the pre-encoder module.

        This method takes an input tensor and its corresponding lengths, 
        processes the input through the pre-encoder network, and returns 
        the encoded output along with the updated lengths.

        Args:
            input (torch.Tensor): The input tensor to be processed, typically of 
                shape (batch_size, sequence_length, input_size).
            input_lengths (torch.Tensor): A tensor containing the lengths of 
                each input sequence in the batch, of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The encoded output tensor, typically of 
                shape (batch_size, output_size).
                - output_lengths (torch.Tensor): A tensor containing the lengths 
                of the output sequences, of shape (batch_size,).

        Raises:
            NotImplementedError: If the method is called directly on the 
                abstract class.

        Examples:
            >>> model = SomeConcretePreEncoder()
            >>> input_tensor = torch.randn(32, 10, 64)  # Example input
            >>> input_lengths = torch.tensor([10] * 32)  # All sequences are full length
            >>> output, output_lengths = model(input_tensor, input_lengths)
            >>> print(output.shape)  # Expected output shape: (32, output_size)
            >>> print(output_lengths.shape)  # Expected output_lengths shape: (32,)

        Note:
            This method must be implemented by any subclass of AbsPreEncoder.
        """
        raise NotImplementedError
