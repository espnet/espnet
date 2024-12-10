from abc import ABC, abstractmethod
from typing import Tuple

import torch

from espnet.nets.scorer_interface import BatchScorerInterface


class AbsLM(torch.nn.Module, BatchScorerInterface, ABC):
    """
    The abstract base class for Language Models (LMs) in ESPnet.

    This class defines the interface for language models and shares the
    loss calculation method among different model implementations. It
    employs the delegate pattern, where an instance of this class is
    passed to the `LanguageModel`.

    Example:
        >>> from espnet2.lm.abs_model import AbsLM
        >>> lm = AbsLM()  # This will raise NotImplementedError
        >>> model = LanguageESPnetModel(lm=lm)

    Note:
        This class cannot be instantiated directly since it is an abstract
        class. Subclasses must implement the `forward` method.

    Attributes:
        None

    Methods:
        forward(input: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Abstract method that must be implemented by subclasses to define
            the forward pass of the language model.

    Raises:
        NotImplementedError: If the `forward` method is called without being
        implemented in a subclass.
    """

    @abstractmethod
    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Computes the forward pass of the language model.

        This method takes an input tensor and a hidden state tensor, processes them
        through the language model, and returns the output tensor along with the
        updated hidden state. It must be implemented by subclasses of the abstract
        class `AbsLM`.

        Args:
            input (torch.Tensor): The input tensor containing the data for the
                language model.
            hidden (torch.Tensor): The hidden state tensor from the previous time
                step, which is updated during the forward pass.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The output tensor after processing the
                  input through the model.
                - hidden (torch.Tensor): The updated hidden state tensor.

        Raises:
            NotImplementedError: If this method is called directly on an instance
                of `AbsLM`, since it must be implemented in subclasses.

        Examples:
            >>> model = MyLanguageModel()
            >>> input_tensor = torch.randn(10, 32)  # (sequence_length, batch_size)
            >>> hidden_state = torch.zeros(1, 32, 256)  # (num_layers, batch_size, hidden_size)
            >>> output, new_hidden = model.forward(input_tensor, hidden_state)

        Note:
            This method is essential for the functioning of the language model and
            should be properly defined in any subclass.
        """
        raise NotImplementedError
