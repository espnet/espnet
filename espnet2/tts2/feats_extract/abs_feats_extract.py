from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class AbsFeatsExtractDiscrete(torch.nn.Module, ABC):
    """
    Abstract base class for extracting discrete features from input sequences.

    This class provides an interface for subclasses to implement methods for
    parsing discrete token sequences into structured data formats suitable for
    prediction tasks. The specific implementations may include options such as
    retaining the sequence as-is, resizing it into a matrix, or applying
    multi-resolution techniques.

    Attributes:
        None

    Args:
        input (torch.Tensor): A tensor containing the input discrete token
            sequence.
        input_lengths (torch.Tensor): A tensor indicating the lengths of the
            input sequences.

    Returns:
        Tuple[Any, Dict]: A tuple containing the processed features and a
            dictionary with additional information.

    Raises:
        NotImplementedError: If the `forward` method is not implemented by
            a subclass.

    Examples:
        >>> model = SomeConcreteFeatsExtractDiscrete()  # An example subclass
        >>> input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> input_lengths = torch.tensor([3, 3])
        >>> features, info = model(input_tensor, input_lengths)

    Note:
        This is an abstract class and cannot be instantiated directly.
        Subclasses must implement the `forward` method.
    """

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[Any, Dict]:
        """
        Forward method for processing a discrete token sequence.

        This method takes an input tensor representing a sequence of discrete
        tokens and their corresponding lengths. It is expected to transform
        the input into a structured format suitable for further prediction tasks.

        Args:
            input (torch.Tensor): A tensor containing the input token sequence.
            input_lengths (torch.Tensor): A tensor containing the lengths of
                each input sequence.

        Returns:
            Tuple[Any, Dict]: A tuple where the first element can be any type
            representing the processed output, and the second element is a
            dictionary containing additional information related to the
            processing.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.

        Examples:
            >>> model = SomeConcreteModel()  # A subclass of AbsFeatsExtractDiscrete
            >>> input_tensor = torch.tensor([[1, 2, 3], [4, 5, 0]])
            >>> lengths = torch.tensor([3, 2])
            >>> output, info = model.forward(input_tensor, lengths)
        """
        raise NotImplementedError
