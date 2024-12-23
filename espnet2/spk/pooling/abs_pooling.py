from abc import ABC, abstractmethod

import torch


class AbsPooling(torch.nn.Module, ABC):
    """
        Abstract base class for pooling layers in the ESPnet2 speaker processing
    module.

    This class inherits from `torch.nn.Module` and serves as a blueprint for
    creating various pooling strategies. It defines the essential methods that
    must be implemented by any concrete pooling class.

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If the derived class does not implement the required
        methods.

    Examples:
        class CustomPooling(AbsPooling):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                # Implement the pooling logic here
                pass

            def output_size(self) -> int:
                # Return the size of the output tensor
                return 0

    Note:
        This class should not be instantiated directly.

    Todo:
        Implement specific pooling strategies by extending this class.
    """

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
                Abstract method to define the forward pass for the AbsPooling class.

        This method must be implemented by subclasses of AbsPooling to specify how
        the pooling operation should be performed on the input tensor.

        Args:
            input (torch.Tensor): The input tensor on which the pooling operation
                will be applied.

        Returns:
            torch.Tensor: The output tensor after applying the pooling operation.

        Raises:
            NotImplementedError: If this method is not implemented by a subclass.

        Examples:
            class MaxPooling(AbsPooling):
                def forward(self, input: torch.Tensor) -> torch.Tensor:
                    return torch.max(input, dim=1)[0]

            pooling_layer = MaxPooling()
            output = pooling_layer.forward(torch.randn(1, 3, 32, 32))
            print(output.shape)  # Example output shape after pooling
        """
        raise NotImplementedError

    @abstractmethod
    def output_size(self) -> int:
        """
                Abstract base class for pooling layers in PyTorch.

        This class defines the interface for pooling layers that can be
        implemented by derived classes. It includes an abstract method for
        the forward pass and an abstract method for determining the output
        size of the pooling operation.

        Attributes:
            None

        Args:
            None

        Returns:
            None

        Yields:
            None

        Raises:
            NotImplementedError: If the method is not implemented in a derived class.

        Examples:
            To create a custom pooling layer, inherit from this class and
            implement the `forward` and `output_size` methods:

            class CustomPooling(AbsPooling):
                def forward(self, input: torch.Tensor) -> torch.Tensor:
                    # Custom pooling logic
                    return pooled_output

                def output_size(self) -> int:
                    # Logic to calculate output size
                    return calculated_output_size

        Note:
            This is an abstract base class and cannot be instantiated directly.
        """
        raise NotImplementedError
