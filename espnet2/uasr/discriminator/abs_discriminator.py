from abc import ABC, abstractmethod

import torch


class AbsDiscriminator(torch.nn.Module, ABC):
    """
        Abstract base class for implementing discriminators in the ESPnet2 framework.

    This class defines the interface for all discriminator implementations. It
    inherits from `torch.nn.Module` and requires subclasses to implement the
    `forward` method. The forward method is responsible for processing input
    tensors and producing output tensors, typically used in adversarial training
    settings.

    Attributes:
        None

    Args:
        xs_pad (torch.Tensor): A padded input tensor containing the features.
        padding_mask (torch.Tensor): A tensor indicating the padding positions
            in `xs_pad`.

    Returns:
        torch.Tensor: The output tensor produced by the discriminator after
            processing the input features.

    Raises:
        NotImplementedError: If the `forward` method is not implemented in a
            subclass.

    Examples:
        To create a custom discriminator, inherit from this class and implement
        the `forward` method as follows:

        ```python
        class MyDiscriminator(AbsDiscriminator):
            def forward(self, xs_pad, padding_mask):
                # Custom processing logic here
                return output_tensor
        ```

    Note:
        This class is intended to be subclassed, and should not be instantiated
        directly.
    """

    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
                Computes the forward pass of the discriminator model.

        This method takes padded input sequences and their corresponding padding masks
        to produce a tensor output. It must be implemented by any subclass of
        `AbsDiscriminator`.

        Attributes:
            xs_pad (torch.Tensor): Padded input sequences for the discriminator.
            padding_mask (torch.Tensor): Mask indicating the positions of padding in
            the input sequences.

        Args:
            xs_pad (torch.Tensor): A tensor of shape (batch_size, seq_length,
            feature_dim) containing the padded input data.
            padding_mask (torch.Tensor): A tensor of shape (batch_size, seq_length)
            indicating which elements of `xs_pad` are valid (1) and which are padding (0).

        Returns:
            torch.Tensor: A tensor containing the output from the discriminator, with
            shape (batch_size, output_dim).

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            >>> discriminator = MyDiscriminator()  # MyDiscriminator should extend AbsDiscriminator
            >>> xs_pad = torch.rand(32, 10, 64)  # Example input (32 samples, seq_len=10, features=64)
            >>> padding_mask = torch.ones(32, 10)  # Example mask (no padding)
            >>> output = discriminator(xs_pad, padding_mask)
            >>> print(output.shape)  # Output shape will depend on the implementation

        Note:
            This method must be overridden in any subclass of `AbsDiscriminator`.
        """
        raise NotImplementedError
