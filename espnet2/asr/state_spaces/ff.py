# This code is derived from https://github.com/HazyResearch/state-spaces

"""Implementation of FFN block in the style of Transformers."""

from functools import partial

from torch import nn

from espnet2.asr.state_spaces.base import SequenceModule
from espnet2.asr.state_spaces.components import DropoutNd, LinearActivation


class FF(SequenceModule):
    """
        Feed-Forward Network (FFN) block in the style of Transformers.

    This class implements a configurable Feed-Forward Network block, commonly used in
    Transformer architectures. It consists of two linear layers with an activation
    function and optional dropout in between.

    Attributes:
        d_output (int): The output dimension of the FFN block.
        transposed (bool): If True, the input is expected to be in transposed form.
        ff (nn.Sequential): The sequential layers that make up the FFN block.

    Args:
        d_input (int): The input dimension.
        expand (int, optional): The expansion factor for the hidden dimension. Defaults to 2.
        d_output (int, optional): The output dimension. If None, it's set to d_input. Defaults to None.
        transposed (bool, optional): Whether the input is transposed. Defaults to False.
        activation (str, optional): The activation function to use. Defaults to "gelu".
        initializer (callable, optional): The initializer for the linear layers. Defaults to None.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        tie_dropout (bool, optional): If True, uses the same dropout mask for all elements. Defaults to False.

    Example:
        >>> ffn = FF(d_input=512, expand=4, d_output=512, dropout=0.1)
        >>> x = torch.randn(32, 100, 512)  # (batch_size, seq_len, d_input)
        >>> output, _ = ffn(x)
        >>> print(output.shape)
        torch.Size([32, 100, 512])

    Note:
        This implementation is derived from the state-spaces repository by Hazy Research.
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
                Forward pass of the Feed-Forward Network.

        This method applies the feed-forward transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor. Shape depends on the 'transposed' attribute:
                - If transposed=False: [batch_size, seq_len, d_input]
                - If transposed=True: [batch_size, d_input, seq_len]
            *args: Variable length argument list. Not used in this method but included for compatibility.
            **kwargs: Arbitrary keyword arguments. Not used in this method but included for compatibility.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after applying the feed-forward transformation.
                  Shape will be the same as the input tensor, with the last dimension changed to d_output.
                - None: Placeholder for consistency with other modules that might return a state.

        Example:
            >>> ffn = FF(d_input=512, d_output=512)
            >>> x = torch.randn(32, 100, 512)  # (batch_size, seq_len, d_input)
            >>> output, _ = ffn.forward(x)
            >>> print(output.shape)
            torch.Size([32, 100, 512])
        """
        return self.ff(x), None

    def step(self, x, state, **kwargs):
        """
                Perform a single step of the Feed-Forward Network.

        This method is designed for sequential processing, applying the feed-forward
        transformation to a single time step of the input.

        Args:
            x (torch.Tensor): The input tensor for a single time step.
                - If transposed=False: Shape is [batch_size, d_input]
                - If transposed=True: Shape is [batch_size, d_input, 1]
            state: Unused parameter, included for compatibility with other sequential modules.
            **kwargs: Additional keyword arguments. Not used in this method but included for compatibility.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after applying the feed-forward transformation.
                  Shape will be [batch_size, d_output] if not transposed, or [batch_size, d_output, 1] if transposed.
                - Any: The unchanged state parameter, returned for consistency with other sequential modules.

        Example:
            >>> ffn = FF(d_input=512, d_output=512, transposed=False)
            >>> x = torch.randn(32, 512)  # (batch_size, d_input)
            >>> output, _ = ffn.step(x, None)
            >>> print(output.shape)
            torch.Size([32, 512])

        Note:
            The behavior of this method depends on the 'transposed' attribute of the FF class.
            When transposed=True, the input is unsqueezed before processing and squeezed afterwards.
        """
        # x: [batch, d_input]
        if self.transposed:
            # expects: [batch, d_input, seq_len]
            return self.ff(x.unsqueeze(-1)).squeeze(-1), state
        else:
            return self.ff(x), state
