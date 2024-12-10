"""Positional encoding modules."""

import math

import torch

from espnet.nets.pytorch_backend.transformer.embedding import _pre_hook


class RelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module for sequence processing.

    This module implements relative positional encoding, which enhances the 
    performance of attention mechanisms in sequence models by providing 
    contextual information about the position of elements in the input 
    sequences.

    Attributes:
        size (int): The dimensionality of the positional encoding.
        pe (torch.Tensor): The computed positional encodings.
        dropout (torch.nn.Dropout): The dropout layer applied to the positional 
            encodings.

    Args:
        size (int): Module size, representing the dimensionality of the 
            positional encoding.
        max_len (int): Maximum length of input sequences for which positional 
            encodings will be computed.
        dropout_rate (float, optional): Dropout rate applied to the output 
            positional encodings. Default is 0.0.

    Methods:
        extend_pe(x: torch.Tensor, left_context: int = 0) -> None:
            Resets the positional encoding based on the input sequences.

        forward(x: torch.Tensor, left_context: int = 0) -> torch.Tensor:
            Computes the positional encoding for the given input sequences.

    Examples:
        # Create a relative positional encoding module
        rpe = RelPositionalEncoding(size=128, dropout_rate=0.1, max_len=5000)

        # Input tensor of shape (B, T, ?)
        input_tensor = torch.randn(32, 100, 128)

        # Get the positional encoding
        pos_enc = rpe(input_tensor, left_context=10)
        print(pos_enc.shape)  # Output shape will be (B, 2 * (T - 1), ?)

    Note:
        The `extend_pe` method should be called before computing the forward 
        pass to ensure the positional encodings are appropriately sized for 
        the input.

    Todo:
        - Add support for variable input sizes beyond the maximum length.
    """

    def __init__(
        self, size: int, dropout_rate: float = 0.0, max_len: int = 5000
    ) -> None:
        """Construct a RelativePositionalEncoding object."""
        super().__init__()

        self.size = size

        self.pe = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x: torch.Tensor, left_context: int = 0) -> None:
        """
        Positional encoding modules.

        This module implements relative positional encoding, which is commonly used 
        in transformer architectures to provide information about the position of 
        tokens in a sequence.

        Attributes:
            size (int): The size of the positional encoding.
            pe (torch.Tensor): The tensor holding the positional encodings.
            dropout (torch.nn.Dropout): The dropout layer applied to the positional 
                encodings.

        Args:
            size (int): Module size.
            max_len (int): Maximum input length.
            dropout_rate (float): Dropout rate.

        Methods:
            extend_pe(x: torch.Tensor, left_context: int = 0) -> None:
                Resets the positional encoding based on the input tensor.

            forward(x: torch.Tensor, left_context: int = 0) -> torch.Tensor:
                Computes the positional encoding for the input tensor.

        Examples:
            # Create an instance of RelPositionalEncoding
            rel_pos_enc = RelPositionalEncoding(size=64, dropout_rate=0.1, max_len=5000)

            # Generate a random input tensor of shape (batch_size, seq_len, features)
            input_tensor = torch.randn(32, 100, 64)

            # Compute the positional encoding
            pos_enc = rel_pos_enc(input_tensor, left_context=10)

        Note:
            The `extend_pe` method is called internally in the `forward` method to 
            ensure the positional encodings are updated based on the input tensor.

        Todo:
            - Add support for different types of input sequences if necessary.
        """
        time1 = x.size(1) + left_context

        if self.pe is not None:
            if self.pe.size(1) >= time1 * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(device=x.device, dtype=x.dtype)
                return

        pe_positive = torch.zeros(time1, self.size)
        pe_negative = torch.zeros(time1, self.size)

        position = torch.arange(0, time1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.size, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.size)
        )

        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)

        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_negative = pe_negative[1:].unsqueeze(0)

        self.pe = torch.cat([pe_positive, pe_negative], dim=1).to(
            dtype=x.dtype, device=x.device
        )

    def forward(self, x: torch.Tensor, left_context: int = 0) -> torch.Tensor:
        """
        Compute positional encoding.

        This method generates the positional encoding for the input sequences, 
        utilizing relative positional encoding to enhance the model's ability 
        to attend to previous elements in the input.

        Args:
            x: Input sequences of shape (B, T, ?), where B is the batch size, 
            T is the sequence length, and ? represents any additional dimensions.
            left_context: Number of previous frames the attention module can see 
                        in the current chunk. This is used to determine the size 
                        of the positional encoding.

        Returns:
            pos_enc: Positional embedding sequences of shape (B, 2 * (T - 1), ?), 
                    which incorporates both positive and negative positional encodings.

        Examples:
            >>> rel_pos_enc = RelPositionalEncoding(size=128)
            >>> input_tensor = torch.randn(10, 20, 128)  # Batch of 10, seq len 20
            >>> output = rel_pos_enc.forward(input_tensor, left_context=5)
            >>> output.shape
            torch.Size([10, 39, 128])  # Output shape will vary based on left_context

        Note:
            The method uses the `extend_pe` function to ensure that the positional 
            encodings are correctly sized for the input sequences before applying 
            the dropout.

        Raises:
            ValueError: If the input tensor `x` does not have the expected shape.
        """
        self.extend_pe(x, left_context=left_context)

        time1 = x.size(1) + left_context

        pos_enc = self.pe[
            :, self.pe.size(1) // 2 - time1 + 1 : self.pe.size(1) // 2 + x.size(1)
        ]
        pos_enc = self.dropout(pos_enc)

        return pos_enc
