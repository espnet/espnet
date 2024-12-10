#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
from typing import Any

import torch


class AbsTokenizer(torch.nn.Module, ABC):
    """
    The abstract tokenizer class for SpeechLM.

    The main objective of this module is to transform the LM-generated tokens
    into the corresponding targets. For example:
        - Speech Codec codes -> waveform
        - BPE tokens -> text

    This class serves as a blueprint for implementing specific tokenizers
    that can handle various types of tokens and their corresponding outputs.

    Attributes:
        None

    Args:
        None

    Returns:
        Any: The output corresponding to the input tokens after processing.

    Yields:
        None

    Raises:
        NotImplementedError: If the forward method is not implemented in a
        derived class.

    Examples:
        class MyTokenizer(AbsTokenizer):
            def forward(self, tokens: torch.Tensor) -> str:
                # Implement specific token transformation logic here
                pass

    Note:
        This is an abstract class, and the `forward` method must be
        implemented in any subclass to define how tokens are processed.
    """

    @abstractmethod
    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> Any:
        """
            Processes the input tokens and transforms them into the corresponding
        targets. This method should be implemented by subclasses of the
        AbsTokenizer class to provide specific tokenization logic.

        Args:
            tokens (torch.Tensor): A tensor containing the input tokens to be
            processed. The shape and type of the tensor will depend on the
            specific tokenizer implementation.

        Returns:
            Any: The output after processing the input tokens. The type and
            structure of the output will vary based on the implementation.

        Raises:
            NotImplementedError: If the method is called directly from the
            AbsTokenizer class, as it is meant to be implemented in a subclass.

        Examples:
            Assuming a subclass `MyTokenizer` implements the forward method:

            >>> tokenizer = MyTokenizer()
            >>> input_tokens = torch.tensor([1, 2, 3])
            >>> output = tokenizer.forward(input_tokens)
            >>> print(output)  # This will print the processed output based on
                             # the specific logic defined in MyTokenizer.

        Note:
            This method is decorated with `@torch.no_grad()` to disable
            gradient calculation, as tokenization does not require
            backpropagation.

        Todo:
            Implement the forward method in subclasses to define specific
            tokenization logic.
        """
        raise NotImplementedError
