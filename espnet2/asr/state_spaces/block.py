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
    Residual block wrapper for a black box layer.

    The `SequenceResidualBlock` class implements a generic
    transformation from (batch, length, d_input) to (batch, length, d_input).
    It provides configurable options for normalization, subsampling, and residual
    connections, making it versatile for various neural network architectures.

    Attributes:
        d_input (int): Input feature dimension.
        i_layer (int, optional): Layer index, required for certain residuals.
        prenorm (bool): If True, applies normalization before the black box layer.
        dropout (float): Dropout rate for the black box module.
        tie_dropout (bool): If True, ties the dropout mask across the sequence.
        transposed (bool): If True, transposes inputs for the black box layer.
        layer (nn.Module): Instance of the black box module.
        residual (nn.Module, optional): Instance of the residual function.
        norm (Normalization, optional): Instance of the normalization layer.
        pool (nn.Module, optional): Instance of the pooling layer.
        drop_path (StochasticDepth): Instance for stochastic depth.

    Args:
        d_input (int): Input feature dimension.
        i_layer (int, optional): Layer index, for certain residuals.
        prenorm (bool): Apply normalization before the black box layer.
        dropout (float): Dropout rate for the black box module.
        tie_dropout (bool): Tie dropout mask across the sequence.
        transposed (bool): Transpose inputs for each layer.
        layer (dict): Config for the black box module.
        residual (dict, optional): Config for the residual function.
        norm (dict, optional): Config for the normalization layer.
        pool (dict, optional): Config for pooling layer per stage.
        drop_path (float): Drop ratio for stochastic depth.

    Returns:
        Tuple[Tensor, Any]: Output tensor and updated state.

    Examples:
        >>> block = SequenceResidualBlock(d_input=128, layer={'type': 'Conv1d'})
        >>> output, state = block(input_tensor, state)

    Note:
        Ensure that the input tensor shape matches (batch, length, d_input).
    
    Raises:
        ValueError: If the input tensor dimensions do not match expected shape.
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
        Residual block wrapper for black box layer.

    This class implements a residual block that wraps around a black box layer,
    allowing for a flexible and configurable transformation of input sequences.
    The block supports various configurations for normalization, pooling, and
    residual connections, enabling advanced architectures in sequence modeling.

    Attributes:
        d_input (int): Input feature dimension.
        i_layer (int, optional): Layer index for certain residuals like Decay.
        prenorm (bool): Indicates if normalization should be applied before 
            the black box layer.
        dropout (float): Dropout rate for the black box module.
        tie_dropout (bool): If True, ties dropout mask across sequence.
        transposed (bool): If True, transposes inputs to (batch, dim, length).
        layer (nn.Module): Configured black box module.
        residual (nn.Module, optional): Configured residual function.
        norm (Normalization, optional): Normalization layer.
        pool (Pooling, optional): Pooling layer per stage.
        drop_path (StochasticDepth, optional): Stochastic depth for dropout.

    Args:
        d_input (int): Input feature dimension.
        i_layer (int, optional): Layer index for specific residual configurations.
        prenorm (bool): If True, applies normalization before the black box layer.
        dropout (float): Dropout probability for the black box module.
        tie_dropout (bool): If True, ties the dropout mask across sequences.
        transposed (bool): If True, transposes inputs for the layer.
        layer (dict, optional): Configuration for the black box module.
        residual (dict, optional): Configuration for the residual function.
        norm (dict, optional): Configuration for normalization.
        pool (dict, optional): Configuration for pooling layer.
        drop_path (float): Drop ratio for stochastic depth.

    Returns:
        tuple: A tuple containing the output tensor and the updated state.

    Examples:
        >>> block = SequenceResidualBlock(d_input=128, dropout=0.1)
        >>> input_tensor = torch.randn(32, 10, 128)  # (batch, length, features)
        >>> output, state = block(input_tensor)

    Note:
        - The block can be configured to use various normalization and 
          residual options.
        - Ensure that the dimensions of the input tensor match d_input.

    Raises:
        ValueError: If the provided configurations are incompatible.
        """
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        """
        Residual state dimension property for the SequenceResidualBlock class.

This property retrieves the state dimension from the underlying black box layer
within the SequenceResidualBlock. The state dimension is typically utilized for 
maintaining the hidden states across time steps in sequential models.

Returns:
    int: The dimension of the state from the black box layer.

Examples:
    >>> block = SequenceResidualBlock(d_input=128, layer={'type': 'some_layer'})
    >>> state_dim = block.d_state
    >>> print(state_dim)
    128  # Assuming the layer's output state dimension is 128

Note:
    This property assumes that the underlying layer has a defined `d_state` 
    attribute. If the layer does not define `d_state`, an AttributeError 
    will be raised.
        """
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        """
        Converts the internal state of the layer to a tensor format.

This property provides a tensor representation of the current state
of the layer, which can be useful for various operations, such as 
logging, visualization, or further processing. The specific format 
of the tensor is determined by the underlying black box layer's 
implementation.

Returns:
    torch.Tensor: A tensor representation of the layer's state.

Examples:
    # Assuming `block` is an instance of SequenceResidualBlock
    state_tensor = block.state_to_tensor
    print(state_tensor.shape)  # Output will depend on the layer's state
        """
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        """
        Return the default state for the black box layer.

    This method serves as a wrapper to retrieve the default state
    from the underlying black box layer. It can be useful for
    initializing the state before processing input data through
    the forward pass.

    Args:
        *args: Variable length argument list to be passed to the 
            underlying layer's default_state method.
        **kwargs: Arbitrary keyword arguments to be passed to the 
            underlying layer's default_state method.

    Returns:
        The default state of the black box layer.

    Examples:
        >>> block = SequenceResidualBlock(d_input=128)
        >>> state = block.default_state()
        >>> print(state)  # Prints the initialized state from the layer

    Note:
        Ensure that the black box layer is properly initialized
        before calling this method, as it relies on the layer's
        configuration.
        """
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, state=None, **kwargs):
        """
        Performs a forward pass through the sequence residual block.

    This method applies the defined transformations to the input tensor `x`.
    It processes the input through normalization (if configured), a black box
    layer, optional residual connections, and pooling (if configured).

    Args:
        x (torch.Tensor): Input tensor of shape (batch, length, d_input).
        state (optional): State information for the black box layer.
        **kwargs: Additional keyword arguments passed to the black box layer.

    Returns:
        tuple: A tuple containing:
            - y (torch.Tensor): Output tensor after transformations.
              Shape is (batch, length, d_output).
            - state: Updated state information from the black box layer.

    Examples:
        >>> block = SequenceResidualBlock(d_input=128)
        >>> input_tensor = torch.randn(32, 10, 128)  # (batch, length, d_input)
        >>> output, new_state = block.forward(input_tensor)

    Note:
        The method applies pre-norm and post-norm based on the configuration.
        If `prenorm` is set to True, normalization is applied before the
        black box layer; otherwise, it is applied afterward.

    Raises:
        ValueError: If the input tensor `x` does not have the expected shape.
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
        Performs a single step of the residual block transformation.

    This method applies the transformation for a single input sample
    in the context of the SequenceResidualBlock. It includes the
    application of normalization, a black box layer, residual
    connections, and pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, length, d_input).
        state (Any): State information from previous layers, which may be
            required for the black box layer.
        **kwargs: Additional keyword arguments to be passed to the black box
            layer.

    Returns:
        tuple: A tuple containing:
            - y (torch.Tensor): The transformed output tensor of shape
              (batch, length, d_output).
            - state (Any): Updated state information after processing the
              input.

    Examples:
        >>> block = SequenceResidualBlock(d_input=128, layer=my_layer_config)
        >>> x = torch.randn(32, 10, 128)  # (batch_size, seq_length, d_input)
        >>> initial_state = block.default_state()
        >>> output, updated_state = block.step(x, initial_state)

    Note:
        Ensure that the dimensions of the input tensor match the expected
        input feature dimension `d_input` defined during initialization.

    Raises:
        ValueError: If the input tensor `x` does not have the expected
        shape or if the state is incompatible with the layer's requirements.
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
