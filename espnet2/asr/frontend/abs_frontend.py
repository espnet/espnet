from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsFrontend(torch.nn.Module, ABC):
    """
    Abstract base class for front-end processing in Automatic Speech Recognition (ASR).

    This class serves as a blueprint for developing various front-end modules in the
    ESPnet2 ASR framework. It inherits from `torch.nn.Module` and defines the required
    methods for subclasses to implement the output size and the forward pass.

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If the output_size or forward method is not implemented
        by the subclass.

    Examples:
        To create a custom frontend, you would inherit from this class and implement
        the required methods:

        ```python
        class CustomFrontend(AbsFrontend):
            def output_size(self) -> int:
                return 128  # Example output size

            def forward(
                self, input: torch.Tensor, input_lengths: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                # Implement the forward pass logic
                processed_input = input  # Placeholder for actual processing
                return processed_input, input_lengths
        ```

    Note:
        Subclasses must implement both the `output_size` and `forward` methods to be
        usable in the ASR pipeline.
    """

    @abstractmethod
    def output_size(self) -> int:
        """
        Abstract method to compute the output size of the frontend module.

        This method should be implemented by any subclass of the AbsFrontend class to
        return the size of the output tensor produced by the forward pass. The output
        size is crucial for ensuring that the dimensions align correctly in the
        subsequent layers of the neural network.

        Returns:
            int: The size of the output tensor.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            class MyFrontend(AbsFrontend):
                def output_size(self) -> int:
                    return 256  # Example output size

            frontend = MyFrontend()
            print(frontend.output_size())  # Output: 256
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the forward pass of the neural network module.

        This method processes the input tensor and its corresponding lengths to
        produce an output tensor, typically used for predictions or further
        processing in the network. The implementation of this method must be
        provided by subclasses of the `AbsFrontend` class.

        Args:
            input (torch.Tensor): The input tensor containing the data to be
                processed. The shape of the tensor should be compatible with
                the model's expected input dimensions.
            input_lengths (torch.Tensor): A tensor containing the lengths of
                each input sequence in the batch. This is necessary for
                processing variable-length sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - The first tensor is the output of the forward pass, which
                may have a shape depending on the specific model architecture.
                - The second tensor can be used for additional information,
                such as hidden states or other relevant data, as defined
                by the specific implementation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            >>> model = MyFrontendSubclass()  # MyFrontendSubclass must implement forward
            >>> input_data = torch.randn(10, 20)  # Example input tensor
            >>> input_lengths = torch.tensor([20] * 10)  # All sequences of length 20
            >>> output, additional_info = model.forward(input_data, input_lengths)

        Note:
            The `forward` method must be overridden in any subclass of
            `AbsFrontend` to provide the specific behavior for processing
            the input data.
        """
        raise NotImplementedError
