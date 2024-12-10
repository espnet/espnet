# This code is derived from https://github.com/HazyResearch/state-spaces

"""Implementation of FFN block in the style of Transformers."""

from functools import partial

from torch import nn

from espnet2.asr.state_spaces.base import SequenceModule
from espnet2.asr.state_spaces.components import DropoutNd, LinearActivation


class FF(SequenceModule):
    """
    Implementation of a Feedforward Neural Network (FFN) block in the style of
    Transformers.

    This class defines a feedforward layer that consists of two linear transformations
    with an activation function in between, along with optional dropout. It can be
    configured to work with both standard and transposed inputs.

    Attributes:
        d_output (int): The dimensionality of the output. If not specified, it
            defaults to the input dimension.
        transposed (bool): Indicates whether the input is in transposed format.

    Args:
        d_input (int): The dimensionality of the input features.
        expand (int): The expansion factor for the inner linear layer. Default is 2.
        d_output (int, optional): The dimensionality of the output features.
            Defaults to None, which sets it to d_input.
        transposed (bool, optional): If True, the input is treated as a
            transposed tensor. Defaults to False.
        activation (str, optional): The activation function to use in the first
            linear layer. Default is "gelu".
        initializer (callable, optional): A function to initialize the weights
            of the linear layers. Defaults to None.
        dropout (float, optional): The dropout probability. Defaults to 0.0 (no
            dropout).
        tie_dropout (bool, optional): If True, ties the dropout for both layers.
            Defaults to False.

    Returns:
        Tuple[torch.Tensor, None]: The output of the feedforward layer and None.

    Examples:
        >>> ff_layer = FF(d_input=256, expand=4, dropout=0.1)
        >>> input_tensor = torch.randn(32, 256)  # [batch_size, d_input]
        >>> output, _ = ff_layer(input_tensor)
        >>> output.shape
        torch.Size([32, 256])

    Note:
        This implementation is derived from the state-spaces library available at
        https://github.com/HazyResearch/state-spaces.

    Todo:
        - Consider adding support for other activation functions.
        - Implement weight initialization strategies based on common practices.
    """

    def __init__(
        self,
        d_input,
        expand=2,
        d_output=None,
        transposed=False,
        activation="gelu",
        initializer=None,
        dropout=0.0,
        tie_dropout=False,
    ):
        super().__init__()
        self.d_output = d_input if d_output is None else d_output
        self.transposed = transposed
        d_inner = expand * d_input

        linear1 = LinearActivation(
            d_input,
            d_inner,
            transposed=transposed,
            activation=activation,
            initializer=initializer,
            activate=True,
        )
        dropout_cls = (
            partial(DropoutNd, transposed=self.transposed)
            if tie_dropout
            else nn.Dropout
        )
        # dropout_cls = nn.Dropout2d if self.transposed else nn.Dropout
        drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        linear2 = LinearActivation(
            d_inner,
            self.d_output,
            transposed=transposed,
            activation=None,
            initializer=initializer,
            activate=False,
        )

        self.ff = nn.Sequential(
            linear1,
            drop,
            linear2,
        )

    def forward(self, x, *args, **kwargs):
        """
        Applies a feed-forward neural network (FFN) transformation to the input data.

        This method is part of the FF class, which implements a feed-forward block
        in the style of Transformers. The forward method processes the input tensor
        through a series of linear transformations, with optional dropout for regularization.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, d_input] or
                [batch_size, d_input, seq_len] if transposed is True.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).

        Returns:
            tuple: A tuple containing the output tensor and None. The output tensor
                has the shape [batch_size, d_output] or [batch_size, d_output, seq_len]
                depending on the transposed flag.

        Examples:
            >>> import torch
            >>> ffn = FF(d_input=128, expand=2, d_output=256)
            >>> input_tensor = torch.randn(32, 128)  # batch_size=32, d_input=128
            >>> output, _ = ffn.forward(input_tensor)
            >>> output.shape
            torch.Size([32, 256])

            >>> ffn_transposed = FF(d_input=128, expand=2, d_output=256, transposed=True)
            >>> input_tensor_transposed = torch.randn(32, 128, 10)  # seq_len=10
            >>> output_transposed, _ = ffn_transposed.forward(input_tensor_transposed)
            >>> output_transposed.shape
            torch.Size([32, 256, 10])

        Note:
            The dropout layer is applied only if the dropout parameter is greater than 0.
        """
        return self.ff(x), None

    def step(self, x, state, **kwargs):
        """
        Executes a forward step through the Feed-Forward (FF) block.

        This method processes the input tensor `x` through the FF block defined in
        the `__init__` method. It applies a series of linear transformations and
        activation functions to the input tensor and returns the result along with
        the current state.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, d_input] or
                [batch, d_input, seq_len] depending on the transposed flag.
            state (Any): The current state to be passed through the network, which is
                not modified by this method.
            **kwargs: Additional keyword arguments that may be used for other
                processing, but are not utilized in this method.

        Returns:
            Tuple[torch.Tensor, Any]: A tuple containing:
                - torch.Tensor: The output tensor after applying the FF block.
                - Any: The unmodified state.

        Examples:
            >>> ff_block = FF(d_input=64, transposed=False)
            >>> input_tensor = torch.randn(32, 64)  # Batch of 32, d_input of 64
            >>> output, state = ff_block.step(input_tensor, state=None)
            >>> print(output.shape)  # Output shape will be [32, d_output]

        Note:
            The output shape depends on whether the `transposed` attribute is set to
            True or False. If `transposed` is True, the input tensor `x` should
            have shape [batch, d_input, seq_len].
        """
        # x: [batch, d_input]
        if self.transposed:
            # expects: [batch, d_input, seq_len]
            return self.ff(x.unsqueeze(-1)).squeeze(-1), state
        else:
            return self.ff(x), state
