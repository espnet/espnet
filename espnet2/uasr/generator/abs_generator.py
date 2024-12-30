from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class AbsGenerator(torch.nn.Module, ABC):
    """
        Abstract base class for generators in the ESPnet2 UASR framework.

    This class defines the essential methods that must be implemented by any
    concrete generator subclass. Generators are responsible for producing outputs
    based on input tensors, and this class provides a standardized interface.

    Attributes:
        None

    Methods:
        output_size() -> int:
            Returns the size of the output tensor produced by the generator.

        forward(xs_pad: torch.Tensor, ilens: torch.Tensor) -> Tuple[torch.Tensor,
        torch.Tensor, Optional[torch.Tensor]]:
            Takes padded input tensors and their lengths, processes them, and
            returns the output tensor along with additional information.

    Raises:
        NotImplementedError:
            If a subclass does not implement the abstract methods.

    Examples:
        class MyGenerator(AbsGenerator):
            def output_size(self) -> int:
                return 256

            def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor) -> Tuple[
                torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                # Implementation goes here
                pass

    Note:
        This class should not be instantiated directly. Instead, subclass it
        to create a specific generator implementation.
    """

    @abstractmethod
    def output_size(self) -> int:
        """
                Returns the size of the output from the generator.

        This property is expected to be implemented by subclasses of `AbsGenerator`
        to provide the specific output size of the generator.

        Returns:
            int: The size of the output produced by the generator.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Examples:
            class MyGenerator(AbsGenerator):
                def output_size(self) -> int:
                    return 256

            gen = MyGenerator()
            print(gen.output_size())  # Output: 256
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
            Performs the forward pass of the generator model.

        This method takes padded input tensors and their corresponding lengths,
        and produces the output tensors as per the generator's architecture.

        Args:
            xs_pad (torch.Tensor): A tensor containing the padded input sequences.
            ilens (torch.Tensor): A tensor containing the lengths of the input
                sequences before padding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - A tensor representing the output of the generator.
                - A tensor representing the hidden states.
                - An optional tensor representing any additional outputs,
                  if applicable.

        Raises:
            NotImplementedError: If this method is not overridden in a subclass.

        Examples:
            >>> generator = MyGenerator()  # MyGenerator is a subclass of AbsGenerator
            >>> xs_pad = torch.tensor([[1, 2, 3], [4, 5, 0]])
            >>> ilens = torch.tensor([3, 2])
            >>> output, hidden, additional = generator.forward(xs_pad, ilens)

        Note:
            The implementation of this method must handle the specific logic
            for generating outputs based on the provided input sequences and
            their lengths.
        """
        raise NotImplementedError
