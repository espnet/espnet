from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class AbsEncoder(torch.nn.Module, ABC):
    """
    Abstract base class for encoders in the ASR (Automatic Speech Recognition) system.

    This class serves as a blueprint for all encoder implementations, ensuring that
    they define the necessary methods for processing input data. Encoders are
    responsible for transforming input sequences into a format suitable for
    further processing in an ASR pipeline.

    Attributes:
        None

    Methods:
        output_size: Returns the size of the output tensor after encoding.
        forward: Processes the input tensor and returns the encoded output along
            with additional state information.

    Args:
        xs_pad (torch.Tensor): Padded input tensor of shape (batch_size, seq_len,
            feature_dim).
        ilens (torch.Tensor): Lengths of the input sequences of shape (batch_size,).
        prev_states (torch.Tensor, optional): Previous hidden states, if applicable.
            Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - Encoded output tensor of shape (batch_size, seq_len, output_dim).
            - Output lengths tensor of shape (batch_size,).
            - Optional: New hidden states tensor if applicable.

    Raises:
        NotImplementedError: If the derived class does not implement the abstract
            methods.

    Examples:
        class MyEncoder(AbsEncoder):
            def output_size(self) -> int:
                return 256

            def forward(self, xs_pad, ilens, prev_states=None):
                # Implementation here
                pass

        encoder = MyEncoder()
        output_size = encoder.output_size()
        encoded_output, output_lengths, new_states = encoder(xs_pad, ilens)

    Note:
        This class is not intended to be instantiated directly. Instead, it should
        be subclassed to create specific encoder implementations.
    """

    @abstractmethod
    def output_size(self) -> int:
        """
        Returns the size of the output produced by the encoder.

        The output size is typically determined by the architecture of the encoder
        and may depend on various factors such as the input size and any learned
        parameters. Implementing classes should provide the specific logic to
        calculate this size based on their unique configurations.

        Returns:
            int: The size of the output tensor produced by the encoder.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Examples:
            class MyEncoder(AbsEncoder):
                def output_size(self) -> int:
                    return 256

            encoder = MyEncoder()
            print(encoder.output_size())  # Output: 256

        Note:
            This method is an abstract method and must be implemented in any
            subclass of AbsEncoder.
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
        Executes the forward pass of the encoder, processing the input tensor and
        producing the output along with the corresponding hidden states.

        This method takes a padded sequence of input tensors, their lengths, and
        optionally the previous hidden states. It outputs a tuple containing the
        encoded representations, the final hidden states, and optionally the
        attention weights if applicable.

        Args:
            xs_pad (torch.Tensor): A tensor of shape (batch_size, seq_len, input_dim)
                containing the padded input sequences.
            ilens (torch.Tensor): A tensor of shape (batch_size,) that contains the
                actual lengths of the input sequences before padding.
            prev_states (torch.Tensor, optional): A tensor of shape (num_layers,
                batch_size, hidden_dim) containing the previous hidden states for
                recurrent architectures. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (batch_size, output_dim) representing the encoded
                outputs.
                - A tensor of shape (num_layers, batch_size, hidden_dim) representing
                the final hidden states.
                - An optional tensor of attention weights, shape (batch_size, seq_len)
                if attention is used, otherwise None.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            >>> encoder = MyEncoder()  # MyEncoder should be a subclass of AbsEncoder
            >>> xs_pad = torch.randn(32, 10, 64)  # Batch of 32, sequence length 10, input dim 64
            >>> ilens = torch.tensor([10] * 32)  # All sequences have length 10
            >>> outputs, hidden_states, _ = encoder(xs_pad, ilens)

        Note:
            The behavior of this method will depend on the specific implementation
            in subclasses of AbsEncoder.
        """
        raise NotImplementedError
