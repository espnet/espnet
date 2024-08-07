from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsPreEncoder(torch.nn.Module, ABC):
    """
        Abstract base class for pre-encoder modules in a neural network.

    This class defines the interface for pre-encoder modules, which are typically
    used to process input data before it's passed to the main encoder in a
    neural network architecture.

    Attributes:
        None

    Examples:
        ```python
        class MyPreEncoder(AbsPreEncoder):
            def output_size(self) -> int:
                return 256

            def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                # Implementation of forward pass
                pass

        pre_encoder = MyPreEncoder()
        ```

    Note:
        Subclasses must implement the `output_size` and `forward` methods.
    """

    @abstractmethod
    def output_size(self) -> int:
        """
                Returns the output size of the pre-encoder.

        This method should be implemented by subclasses to specify the
        dimensionality of the output produced by the pre-encoder.

        Returns:
            int: The size of the output tensor produced by the pre-encoder.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Examples:
            ```python
            class MyPreEncoder(AbsPreEncoder):
                def output_size(self) -> int:
                    return 256

            pre_encoder = MyPreEncoder()
            output_dim = pre_encoder.output_size()  # Returns 256
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Performs the forward pass of the pre-encoder.

        This method should be implemented by subclasses to define the forward
        computation of the pre-encoder.

        Args:
            input (torch.Tensor): The input tensor to be processed.
            input_lengths (torch.Tensor): A tensor containing the lengths of each
                sequence in the input batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The output tensor after pre-encoding.
                - A tensor of output lengths corresponding to each sequence in the batch.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Examples:
            ```python
            class MyPreEncoder(AbsPreEncoder):
                def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    # Example implementation
                    output = self.some_processing(input)
                    output_lengths = input_lengths  # Assuming no change in sequence length
                    return output, output_lengths

            pre_encoder = MyPreEncoder()
            input_tensor = torch.randn(32, 100, 80)  # Batch size 32, 100 time steps, 80 features
            input_lengths = torch.full((32,), 100)  # All sequences in batch have length 100
            output, output_lengths = pre_encoder(input_tensor, input_lengths)
            ```
        """
        raise NotImplementedError
