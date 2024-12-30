# Copyright 2023 Jee-weon Jung
# Apache 2.0

"""RawNet3 Encoder"""

import torch

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class IdentityEncoder(AbsEncoder):
    """
        Identity encoder. Does nothing, just passes frontend feature to the pooling.

    This encoder is expected to be used for cases when the frontend already has a
    good representation, such as self-supervised learning (SSL) features. It simply
    forwards the input tensor without any modifications.

    Attributes:
        _output_size (int): The output feature dimension, which is the same as the
            input feature dimension.

    Args:
        input_size (int): Input feature dimension.

    Returns:
        torch.Tensor: The input tensor transposed along the specified dimensions.

    Examples:
        >>> encoder = IdentityEncoder(input_size=128)
        >>> input_tensor = torch.randn(32, 128, 10)  # (batch_size, input_size, time)
        >>> output_tensor = encoder.forward(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 10, 128])  # Output shape after transposition

    Note:
        The forward method transposes the input tensor from shape (batch_size,
        input_size, time) to (batch_size, time, input_size).
    """

    def __init__(
        self,
        input_size: int,
    ):
        super().__init__()
        self._output_size = input_size

    def output_size(self) -> int:
        """
            Returns the output size of the encoder, which is equal to the input size.

        This property provides the dimension of the features that the encoder
        outputs. It is particularly useful for ensuring compatibility with
        subsequent layers in a neural network.

        Returns:
            int: The size of the output features.

        Examples:
            encoder = IdentityEncoder(input_size=128)
            size = encoder.output_size()  # size will be 128
        """
        return self._output_size

    def forward(self, x: torch.Tensor):
        """
            Passes the input tensor through without modification, transposing its
        dimensions.

        This method is primarily intended for use in scenarios where the input
        features are already adequately represented and require no further
        processing. It simply transposes the input tensor from shape
        (batch_size, input_size, seq_length) to (batch_size, seq_length,
        input_size).

        Args:
            x (torch.Tensor): The input tensor to be processed. It should have
                shape (batch_size, input_size, seq_length).

        Returns:
            torch.Tensor: The transposed tensor with shape
                (batch_size, seq_length, input_size).

        Examples:
            >>> encoder = IdentityEncoder(input_size=128)
            >>> input_tensor = torch.randn(32, 128, 50)  # Example input
            >>> output_tensor = encoder.forward(input_tensor)
            >>> output_tensor.shape
            torch.Size([32, 50, 128])
        """
        return x.transpose(1, 2)
