from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsFrontend(torch.nn.Module, ABC):
    """
        Abstract base class for frontend modules in a neural network.

    This class defines the interface for frontend modules, which are typically
    used in speech processing tasks to convert raw input signals into
    feature representations.

    Attributes:
        None

    Examples:
        >>> class MyFrontend(AbsFrontend):
        ...     def output_size(self) -> int:
        ...         return 80
        ...     def forward(self, input: torch.Tensor, input_lengths: torch.Tensor):
        ...         # Implementation here
        ...         return features, feature_lengths

    Note:
        Subclasses must implement the `output_size` and `forward` methods.
    """

    @abstractmethod
    def output_size(self) -> int:
        """
                Returns the output size of the frontend module.

        This method should be implemented by subclasses to specify the
        dimensionality of the feature representation produced by the frontend.

        Returns:
            int: The size of the output feature dimension.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Examples:
            >>> class MyFrontend(AbsFrontend):
            ...     def output_size(self) -> int:
            ...         return 80
            >>> frontend = MyFrontend()
            >>> print(frontend.output_size())
            80
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Processes the input and returns the feature representation.

        This method should be implemented by subclasses to define how the input
        is transformed into features.

        Args:
            input (torch.Tensor): The input tensor to be processed.
            input_lengths (torch.Tensor): The lengths of each sequence in the input batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - features (torch.Tensor): The processed features.
                - feature_lengths (torch.Tensor): The lengths of each sequence in the feature batch.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Examples:
            >>> class MyFrontend(AbsFrontend):
            ...     def forward(self, input: torch.Tensor, input_lengths: torch.Tensor):
            ...         # Assume some processing here
            ...         features = input * 2
            ...         feature_lengths = input_lengths
            ...         return features, feature_lengths
            >>> frontend = MyFrontend()
            >>> input_tensor = torch.randn(32, 1000)  # Batch of 32, sequence length 1000
            >>> input_lengths = torch.full((32,), 1000)
            >>> features, feature_lengths = frontend(input_tensor, input_lengths)
            >>> features.shape
            torch.Size([32, 1000])
        """
        raise NotImplementedError
