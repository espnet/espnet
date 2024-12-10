from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsNormalize(torch.nn.Module, ABC):
    """
        Abstract base class for normalization layers in the ESPnet framework.

    This class defines the interface for normalization layers that will inherit
    from it. The derived classes must implement the `forward` method to specify
    the normalization behavior. The `forward` method takes an input tensor and
    optionally an input lengths tensor, and it returns the normalized output
    tensor along with its lengths.

    Attributes:
        None

    Args:
        input (torch.Tensor): The input tensor to be normalized.
        input_lengths (torch.Tensor, optional): The lengths of the input tensor.
            Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the normalized output
        tensor and its corresponding lengths tensor.

    Raises:
        NotImplementedError: If the derived class does not implement the `forward`
        method.

    Examples:
        class MyNormalize(AbsNormalize):
            def forward(self, input: torch.Tensor, input_lengths: torch.Tensor = None):
                # Implement normalization logic here
                return output, output_lengths

        normalizer = MyNormalize()
        output, lengths = normalizer(torch.randn(10, 20), torch.tensor([20] * 10))
    """

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Executes the forward pass of the normalization process.

        This method must be implemented by subclasses of `AbsNormalize`. It takes an
        input tensor and an optional tensor of input lengths, and returns the
        normalized output tensor along with the lengths of the output.

        Args:
            input (torch.Tensor): The input tensor to be normalized.
            input_lengths (torch.Tensor, optional): The lengths of the input
                sequences. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the normalized output
                tensor and the corresponding output lengths.

        Raises:
            NotImplementedError: If this method is called directly without being
                overridden in a subclass.

        Examples:
            >>> model = SomeNormalizationModel()  # Assuming SomeNormalizationModel
            >>> input_tensor = torch.randn(10, 5)  # Example input
            >>> lengths = torch.tensor([5] * 10)  # Example lengths
            >>> output, output_lengths = model.forward(input_tensor, lengths)
        """
        # return output, output_lengths
        raise NotImplementedError
