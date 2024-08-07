from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class AbsEncoder(torch.nn.Module, ABC):
    """
        Abstract base class for encoders in a neural network.

    This class defines the interface for encoders used in various neural network
    architectures. It inherits from both torch.nn.Module and ABC (Abstract Base Class),
    ensuring that it can be used as a PyTorch module while also enforcing the
    implementation of abstract methods in derived classes.

    Attributes:
        None

    Note:
        This class should not be instantiated directly. Instead, it should be
        subclassed with concrete implementations of the abstract methods.

    Examples:
        ```python
        class ConcreteEncoder(AbsEncoder):
            def output_size(self) -> int:
                return 256

            def forward(
                self,
                xs_pad: torch.Tensor,
                ilens: torch.Tensor,
                prev_states: torch.Tensor = None,
            ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                # Implementation details
                pass
        ```
    """

    @abstractmethod
    def output_size(self) -> int:
        """
                Returns the output size of the encoder.

        This abstract method should be implemented by subclasses to specify
        the size of the encoder's output.

        Returns:
            int: The size of the encoder's output.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Example:
            ```python
            class ConcreteEncoder(AbsEncoder):
                def output_size(self) -> int:
                    return 256  # Example output size

            encoder = ConcreteEncoder()
            size = encoder.output_size()  # Returns 256
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
                Performs the forward pass of the encoder.

        This abstract method should be implemented by subclasses to define the
        forward pass of the encoder.

        Args:
            xs_pad (torch.Tensor): Padded input tensor.
            ilens (torch.Tensor): Input lengths tensor.
            prev_states (torch.Tensor, optional): Previous encoder states. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - Encoded output tensor
                - Output lengths tensor
                - Optional updated encoder states

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Example:
            ```python
            class ConcreteEncoder(AbsEncoder):
                def forward(
                    self,
                    xs_pad: torch.Tensor,
                    ilens: torch.Tensor,
                    prev_states: torch.Tensor = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                    # Example implementation
                    encoded_output = self.encode(xs_pad)
                    output_lengths = self.calculate_lengths(ilens)
                    new_states = self.update_states(prev_states)
                    return encoded_output, output_lengths, new_states

            encoder = ConcreteEncoder()
            output, lengths, states = encoder.forward(input_tensor, input_lengths)
            ```
        """
        raise NotImplementedError
