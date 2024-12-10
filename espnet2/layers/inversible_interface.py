from abc import ABC, abstractmethod
from typing import Tuple

import torch


class InversibleInterface(ABC):
    """
        InversibleInterface is an abstract base class that defines the interface for
    inversible layers. Any concrete implementation of this interface must provide
    an implementation for the `inverse` method, which is responsible for computing
    the inverse of a given input tensor.

    Attributes:
        None

    Args:
        input (torch.Tensor): The input tensor to be inverted.
        input_lengths (torch.Tensor, optional): The lengths of the input tensor.
            Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the inverted output
        tensor and the corresponding output lengths.

    Raises:
        NotImplementedError: If the `inverse` method is called without an
        implementation in a subclass.

    Examples:
        class MyInversibleLayer(InversibleInterface):
            def inverse(self, input: torch.Tensor, input_lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
                # Implementation of the inverse operation
                output = input.flip(dims=[-1])  # Example operation
                output_lengths = input_lengths  # Example lengths
                return output, output_lengths

        layer = MyInversibleLayer()
        input_tensor = torch.randn(10, 5)  # Example input
        input_lengths = torch.tensor([5] * 10)  # Example lengths
        output_tensor, output_lengths = layer.inverse(input_tensor, input_lengths)
    """

    @abstractmethod
    def inverse(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Computes the inverse of the given input tensor.

        This method must be implemented by any subclass of the
        InversibleInterface. The implementation should define how to compute the
        inverse transformation of the input tensor, along with the lengths of the
        output.

        Args:
            input (torch.Tensor): The input tensor to be inverted.
            input_lengths (torch.Tensor, optional): The lengths of the input
                sequences. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output tensor
            and the output lengths. The output tensor is the result of the inverse
            transformation applied to the input tensor.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            class MyInversible(InversibleInterface):
                def inverse(self, input: torch.Tensor,
                            input_lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
                    # Example implementation of the inverse method
                    output = input.flip(dims=[-1])  # Flipping the tensor for demo
                    output_lengths = input_lengths  # Assuming lengths remain the same
                    return output, output_lengths

            my_inverse = MyInversible()
            input_tensor = torch.tensor([[1, 2], [3, 4]])
            lengths = torch.tensor([2, 2])
            output_tensor, output_lengths = my_inverse.inverse(input_tensor, lengths)
        """
        # return output, output_lengths
        raise NotImplementedError
