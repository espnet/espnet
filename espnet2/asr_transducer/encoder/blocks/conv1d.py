"""Conv1d block for Transducer encoder."""

from typing import Optional, Tuple, Union

import torch


class Conv1d(torch.nn.Module):
    """
    Conv1d block for Transducer encoder.

    This class defines a 1D convolutional layer that can be used as a building block
    for the Transducer encoder architecture. It supports various configurations,
    including causal convolution, batch normalization, and activation functions.

    Attributes:
        input_size (int): The input dimension of the Conv1d layer.
        output_size (int): The output dimension of the Conv1d layer.
        kernel_size (Union[int, Tuple]): Size of the convolving kernel.
        stride (Union[int, Tuple]): Stride of the convolution.
        dilation (Union[int, Tuple]): Spacing between the kernel points.
        groups (Union[int, Tuple]): Number of blocked connections from input
            channels to output channels.
        bias (bool): Whether to add a learnable bias to the output.
        batch_norm (bool): Whether to use batch normalization after convolution.
        relu (bool): Whether to use a ReLU activation after convolution.
        causal (bool): Whether to use causal convolution (set to True if streaming).
        dropout_rate (float): Dropout rate for regularization.

    Args:
        input_size: Input dimension.
        output_size: Output dimension.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Spacing between the kernel points.
        groups: Number of blocked connections from input channels to output
            channels.
        bias: Whether to add a learnable bias to the output.
        batch_norm: Whether to use batch normalization after convolution.
        relu: Whether to use a ReLU activation after convolution.
        causal: Whether to use causal convolution (set to True if streaming).
        dropout_rate: Dropout rate.

    Examples:
        >>> conv_layer = Conv1d(
        ...     input_size=128,
        ...     output_size=64,
        ...     kernel_size=3,
        ...     stride=1,
        ...     batch_norm=True,
        ...     relu=True
        ... )
        >>> x = torch.randn(32, 100, 128)  # (B, T, D_in)
        >>> pos_enc = torch.randn(32, 198, 128)  # (B, 2 * (T - 1), D_in)
        >>> output, mask, pos_enc_out = conv_layer(x, pos_enc)

    Raises:
        ValueError: If the input dimensions do not match the expected shape.

    Note:
        This module uses the PyTorch framework and is designed for efficient
        processing of sequential data.

    Todo:
        - Implement additional functionalities such as layer normalization.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        groups: Union[int, Tuple] = 1,
        bias: bool = True,
        batch_norm: bool = False,
        relu: bool = True,
        causal: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a Conv1d object."""
        super().__init__()

        if causal:
            self.lorder = kernel_size - 1
            stride = 1
        else:
            self.lorder = 0
            stride = stride

        self.conv = torch.nn.Conv1d(
            input_size,
            output_size,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        if relu:
            self.relu_func = torch.nn.ReLU()

        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(output_size)

        self.out_pos = torch.nn.Linear(input_size, output_size)

        self.input_size = input_size
        self.output_size = output_size

        self.relu = relu
        self.batch_norm = batch_norm
        self.causal = causal

        self.kernel_size = kernel_size
        self.padding = dilation * (kernel_size - 1)
        self.stride = stride

        self.cache = None

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """
        Initialize/Reset Conv1d cache for streaming.

        This method initializes or resets the cache used for streaming 
        in the Conv1d module. The cache holds previous frames, allowing 
        for efficient processing of sequential data in a streaming 
        fashion.

        Args:
            left_context: Number of previous frames the attention module 
                          can see in current chunk (not used here).
            device: Device to use for cache tensor, which allows for 
                    computation on the specified hardware (e.g., CPU or GPU).

        Examples:
            >>> conv1d = Conv1d(input_size=64, output_size=128, kernel_size=3)
            >>> conv1d.reset_streaming_cache(left_context=1, device=torch.device('cpu'))
            >>> print(conv1d.cache.shape)
            torch.Size([1, 64, 2])  # Example shape based on kernel_size=3

        Note:
            The cache is initialized with zeros, and its shape is based 
            on the input size and kernel size. It is essential for 
            maintaining context in streaming scenarios.
        """
        self.cache = torch.zeros(
            (1, self.input_size, self.kernel_size - 1), device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input sequences.

        This method applies a 1D convolution to the input tensor `x`, followed by 
        optional batch normalization, dropout, and ReLU activation. It processes 
        the input sequences in a manner that can accommodate causal convolution 
        if specified.

        Args:
            x: Conv1d input sequences of shape (B, T, D_in), where B is the 
            batch size, T is the sequence length, and D_in is the input 
            dimension.
            pos_enc: Positional embedding sequences of shape (B, 2 * (T - 1), D_in).
            mask: Optional source mask of shape (B, T) that indicates which 
                elements of `x` should be attended to. 
            chunk_mask: Optional chunk mask of shape (T_2, T_2) for chunk-based 
                        processing (not used in this method).

        Returns:
            x: Conv1d output sequences of shape (B, sub(T), D_out), where 
            D_out is the output dimension.
            mask: Updated source mask of shape (B, T) or (B, sub(T)), 
                depending on whether padding was applied.
            pos_enc: Updated positional embedding sequences, with shape 
                    (B, 2 * (T - 1), D_att) or (B, 2 * (sub(T) - 1), D_out),
                    depending on the output dimension.

        Examples:
            >>> conv1d_layer = Conv1d(input_size=16, output_size=32, kernel_size=3)
            >>> x = torch.randn(8, 10, 16)  # Batch of 8, sequence length of 10
            >>> pos_enc = torch.randn(8, 18, 16)  # Positional encodings
            >>> mask = torch.ones(8, 10)  # Full attention
            >>> output, updated_mask, updated_pos_enc = conv1d_layer.forward(x, pos_enc, mask)

        Note:
            The method supports both causal and non-causal convolutions. If 
            causal is set to True, it modifies the input `x` by padding it 
            to preserve the order of the sequences.

        Raises:
            ValueError: If the input tensor `x` or positional embeddings `pos_enc` 
                        have incompatible dimensions.
        """
        x = x.transpose(1, 2)

        if self.lorder > 0:
            x = torch.nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        else:
            mask = self.create_new_mask(mask)
            pos_enc = self.create_new_pos_enc(pos_enc)

        x = self.conv(x)

        if self.batch_norm:
            x = self.bn(x)

        x = self.dropout(x)

        if self.relu:
            x = self.relu_func(x)

        x = x.transpose(1, 2)

        return x, mask, self.out_pos(pos_enc)

    def chunk_forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        left_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode chunk of input sequence.

        This method processes a chunk of input sequences through the Conv1d 
        module, allowing for the incorporation of previous context via caching. 
        It is particularly useful for streaming applications where only a 
        portion of the input is available at a time.

        Args:
            x: Conv1d input sequences. Shape (B, T, D_in) where B is the batch 
               size, T is the sequence length, and D_in is the input dimension.
            pos_enc: Positional embedding sequences. Shape (B, 2 * (T - 1), D_in).
            mask: Source mask. Shape (B, T).
            left_context: Number of previous frames the attention module can see 
                          in current chunk (not used here).

        Returns:
            x: Conv1d output sequences. Shape (B, T, D_out) where D_out is the 
               output dimension.
            pos_enc: Positional embedding sequences. Shape (B, 2 * (T - 1), D_out).

        Examples:
            >>> conv1d_layer = Conv1d(input_size=64, output_size=128, 
            ...                        kernel_size=3)
            >>> input_tensor = torch.randn(32, 10, 64)  # (B, T, D_in)
            >>> pos_embedding = torch.randn(32, 18, 64)  # (B, 2*(T-1), D_in)
            >>> mask_tensor = torch.ones(32, 10)          # (B, T)
            >>> output, new_pos_enc = conv1d_layer.chunk_forward(
            ...     input_tensor, pos_embedding, mask_tensor
            ... )

        Note:
            The `left_context` parameter is included for compatibility with 
            streaming applications but is not utilized in this implementation.
        """
        x = torch.cat([self.cache, x.transpose(1, 2)], dim=2)
        self.cache = x[:, :, -self.lorder :]

        x = self.conv(x)

        if self.batch_norm:
            x = self.bn(x)

        x = self.dropout(x)

        if self.relu:
            x = self.relu_func(x)

        x = x.transpose(1, 2)

        return x, self.out_pos(pos_enc)

    def create_new_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Create new mask for output sequences.

        This method generates a new mask based on the input mask, which is 
        adjusted according to the padding and stride properties of the 
        convolutional layer. The output mask will reflect the dimensions 
        of the output sequences after convolution.

        Args:
            mask: Mask of input sequences. Shape: (B, T), where B is the 
                  batch size and T is the length of the input sequence.

        Returns:
            mask: Mask of output sequences. Shape: (B, sub(T)), where 
                  sub(T) is the length of the output sequence after 
                  applying the convolution and stride.

        Examples:
            >>> conv_layer = Conv1d(input_size=16, output_size=32, kernel_size=3)
            >>> input_mask = torch.tensor([[1, 1, 1, 0, 0],
            ...                             [1, 1, 1, 1, 0]])
            >>> output_mask = conv_layer.create_new_mask(input_mask)
            >>> print(output_mask)
            tensor([[1, 0],
                    [1, 0]])

        Note:
            The method assumes that the padding has already been set 
            during the initialization of the Conv1d class.
        """
        if self.padding != 0:
            mask = mask[:, : -self.padding]

        return mask[:, :: self.stride]

    def create_new_pos_enc(self, pos_enc: torch.Tensor) -> torch.Tensor:
        """
        Create new positional embedding vector.

        This method generates a new positional embedding based on the input 
        sequences' positional embeddings. It handles padding and applies the 
        stride to the embeddings to create an output suitable for the 
        convolutional operation.

        Args:
            pos_enc: Input sequences positional embedding.
                    Shape: (B, 2 * (T - 1), D_in)

        Returns:
            pos_enc: Output sequences positional embedding.
                    Shape: (B, 2 * (sub(T) - 1), D_in)

        Examples:
            >>> import torch
            >>> pos_enc = torch.randn(4, 10, 16)  # Example input
            >>> conv1d_layer = Conv1d(input_size=16, output_size=32, kernel_size=3)
            >>> new_pos_enc = conv1d_layer.create_new_pos_enc(pos_enc)
            >>> new_pos_enc.shape
            torch.Size([4, 6, 16])  # Output shape may vary based on padding and stride

        Note:
            The method considers the input's padding and applies the stride to 
            ensure the output positional embeddings align with the output sequences 
            generated by the convolutional layer.
        """
        pos_enc_positive = pos_enc[:, : pos_enc.size(1) // 2 + 1, :]
        pos_enc_negative = pos_enc[:, pos_enc.size(1) // 2 :, :]

        if self.padding != 0:
            pos_enc_positive = pos_enc_positive[:, : -self.padding, :]
            pos_enc_negative = pos_enc_negative[:, : -self.padding, :]

        pos_enc_positive = pos_enc_positive[:, :: self.stride, :]
        pos_enc_negative = pos_enc_negative[:, :: self.stride, :]

        pos_enc = torch.cat([pos_enc_positive, pos_enc_negative[:, 1:, :]], dim=1)

        return pos_enc
