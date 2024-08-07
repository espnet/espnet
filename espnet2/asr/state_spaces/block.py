# This code is derived from https://github.com/HazyResearch/state-spaces

"""Implements a full residual block around a black box layer.

Configurable options include:
normalization position: prenorm or postnorm
normalization type: batchnorm, layernorm etc.
subsampling/pooling
residual options: feedforward, residual, affine scalars, depth-dependent scaling, etc.
"""

from functools import partial

from torch import nn

import espnet2.asr.state_spaces.utils as utils
from espnet2.asr.state_spaces import registry
from espnet2.asr.state_spaces.base import SequenceModule
from espnet2.asr.state_spaces.components import (
    DropoutNd,
    Normalization,
    StochasticDepth,
)
from espnet2.asr.state_spaces.pool import registry as pool_registry
from espnet2.asr.state_spaces.residual import registry as residual_registry


class SequenceResidualBlock(SequenceModule):
    """
        Residual block wrapper for black box layer.

    This class implements a generic (batch, length, d_input) -> (batch, length, d_input) transformation
    with configurable normalization, pooling, and residual options.

    Attributes:
        i_layer (int): Layer index, used by certain residual functions like Decay.
        d_input (int): Input feature dimension.
        layer (SequenceModule): Black box layer module.
        prenorm (bool): If True, apply normalization before the layer; otherwise, after.
        transposed (bool): If True, transpose inputs so each layer receives (batch, dim, length).
        residual (ResidualFunction): Residual function module.
        norm (Normalization): Normalization layer.
        pool (PoolModule): Pooling layer.
        drop (nn.Module): Dropout module.
        drop_path (nn.Module): Stochastic depth module.

    Args:
        d_input (int): Input feature dimension.
        i_layer (int, optional): Layer index. Defaults to None.
        prenorm (bool, optional): Whether to apply normalization before the layer. Defaults to True.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        tie_dropout (bool, optional): Whether to tie dropout mask across sequence. Defaults to False.
        transposed (bool, optional): Whether to transpose inputs. Defaults to False.
        layer (dict, optional): Config for black box module. Defaults to None.
        residual (dict, optional): Config for residual function. Defaults to None.
        norm (str or dict, optional): Config for normalization layer. Defaults to None.
        pool (dict, optional): Config for pooling layer. Defaults to None.
        drop_path (float, optional): Drop ratio for stochastic depth. Defaults to 0.0.

    Example:
        >>> block = SequenceResidualBlock(d_input=256, prenorm=True, dropout=0.1)
        >>> x = torch.randn(32, 100, 256)  # (batch, length, d_input)
        >>> y, _ = block(x)
        >>> y.shape
        torch.Size([32, 100, 256])

    Note:
        The class supports various configurations for normalization, residual connections,
        pooling, and regularization, making it highly flexible for different architectural designs.
    """

    def __init__(
        self,
        d_input,
        i_layer=None,
        prenorm=True,
        dropout=0.0,
        tie_dropout=False,
        transposed=False,
        layer=None,
        residual=None,
        norm=None,
        pool=None,
        drop_path=0.0,
    ):
        super().__init__()

        self.i_layer = i_layer
        self.d_input = d_input
        # self.layer = utils.instantiate(registry.layer, layer, d_input)
        if layer is None:
            layer = {}
        self.layer = utils.instantiate(registry.layer, layer, d_input)
        self.prenorm = prenorm
        self.transposed = transposed

        # Residual
        # d_residual is the output dimension after residual
        if residual is None:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            self.residual = utils.instantiate(
                residual_registry, residual, i_layer, d_input, self.layer.d_output
            )
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate
        # since Normalization has some special cases
        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)

        # Pool
        self.pool = utils.instantiate(
            pool_registry, pool, self.d_residual, transposed=self.transposed
        )

        # Dropout
        dropout_cls = (
            partial(DropoutNd, transposed=self.transposed)
            if tie_dropout
            else nn.Dropout
        )
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        # Stochastic depth
        self.drop_path = (
            StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()
        )

    @property
    def d_output(self):
        """
                Output dimension of the SequenceResidualBlock.

        Returns:
            int: The output dimension. If a pooling layer is present, it returns the output
                 dimension of the pooling layer. Otherwise, it returns the residual dimension.

        Note:
            This property dynamically computes the output dimension based on the block's
            configuration, taking into account whether pooling is applied.
        """
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        """
                State dimension of the underlying layer.

        Returns:
            int: The state dimension of the black box layer.

        Note:
            This property provides access to the state dimension of the internal layer,
            which is useful for understanding the hidden state size in stateful models.
        """
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        """
                Method to convert the layer's state to a tensor.

        Returns:
            callable: The state_to_tensor method of the underlying layer.

        Note:
            This property provides access to the state_to_tensor method of the internal layer,
            allowing for consistent state handling across the residual block wrapper.
        """
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        """
                Get the default state for the underlying layer.

        This method delegates to the default_state method of the internal layer.

        Args:
            *args: Variable length argument list to be passed to the layer's default_state method.
            **kwargs: Arbitrary keyword arguments to be passed to the layer's default_state method.

        Returns:
            The default state of the underlying layer.

        Note:
            This method allows the SequenceResidualBlock to maintain the same interface
            for state initialization as its internal layer.
        """
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, state=None, **kwargs):
        """
                Forward pass of the SequenceResidualBlock.

        This method applies the full sequence of operations: normalization (if prenorm),
        black box layer, residual connection, normalization (if postnorm), and pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, d_input).
            state (Any, optional): Initial state for the layer. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the black box layer.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor after all operations.
                - Any: Updated state of the black box layer.

        Example:
            >>> block = SequenceResidualBlock(d_input=256)
            >>> x = torch.randn(32, 100, 256)
            >>> output, new_state = block(x)
            >>> output.shape
            torch.Size([32, 100, 256])

        Note:
            The order of operations and the application of each component (norm, residual, pool)
            depends on the block's configuration.
        """
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm(y)

        # Black box layer
        y, state = self.layer(y, state=state, **kwargs)

        # Residual
        if self.residual is not None:
            y = self.residual(x, self.drop_path(self.drop(y)), self.transposed)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm(y)

        # Pool
        if self.pool is not None:
            y = self.pool(y)

        return y, state

    def step(self, x, state, **kwargs):
        """
                Perform a single step forward pass of the SequenceResidualBlock.

        This method is designed for sequential processing, applying the block's operations
        on a single time step input.

        Args:
            x (torch.Tensor): Input tensor for a single time step.
            state (Any): Current state of the layer.
            **kwargs: Additional keyword arguments to be passed to the black box layer's step method.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor after all operations for the current time step.
                - Any: Updated state of the black box layer.

        Note:
            This method follows a similar sequence of operations as the forward method,
            but is adapted for step-by-step processing. It does not apply dropout or stochastic depth,
            and the residual connection is applied without transposition.

        Example:
            >>> block = SequenceResidualBlock(d_input=256)
            >>> x = torch.randn(32, 256)  # Single time step input
            >>> state = block.default_state(batch_size=32)
            >>> output, new_state = block.step(x, state)
            >>> output.shape
            torch.Size([32, 256])
        """
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm.step(y)

        # Black box layer
        y, state = self.layer.step(y, state, **kwargs)

        # Residual
        if self.residual is not None:
            y = self.residual(
                x, y, transposed=False
            )  # NOTE this would not work with concat residual function (catformer)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm.step(y)

        # Pool
        if self.pool is not None:
            y = self.pool(y)

        return y, state
