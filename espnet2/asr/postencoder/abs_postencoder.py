from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsPostEncoder(torch.nn.Module, ABC):
    """
        Abstract base class for post-encoder modules in a neural network.

    This class defines the interface for post-encoder modules that process
    encoded input sequences. It inherits from both torch.nn.Module and ABC
    (Abstract Base Class).

    Attributes:
        None

    Note:
        Subclasses must implement the `output_size` and `forward` methods.

    Example:
        class CustomPostEncoder(AbsPostEncoder):
            def output_size(self) -> int:
                return 256

            def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                # Custom implementation
                pass
    """

    @abstractmethod
    def output_size(self) -> int:
        """
                Returns the output size of the post-encoder.

        This abstract method should be implemented by subclasses to specify
        the size of the output tensor produced by the post-encoder.

        Returns:
            int: The size of the output tensor.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Example:
            class CustomPostEncoder(AbsPostEncoder):
                def output_size(self) -> int:
                    return 256  # Example output size
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Processes the input tensor through the post-encoder.

        This abstract method should be implemented by subclasses to define the
        forward pass of the post-encoder.

        Args:
            input (torch.Tensor): The input tensor to be processed.
            input_lengths (torch.Tensor): The lengths of each sequence in the input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The processed output tensor.
                - The updated lengths of each sequence in the output tensor.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Example:
            class CustomPostEncoder(AbsPostEncoder):
                def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    # Example implementation
                    processed_output = some_processing(input)
                    updated_lengths = input_lengths  # Or modify if needed
                    return processed_output, updated_lengths
        """
        raise NotImplementedError
