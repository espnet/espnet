# This code is derived from https://github.com/HazyResearch/state-spaces

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract


def stochastic_depth(input: torch.tensor, p: float, mode: str, training: bool = True):
    """
    Apply stochastic depth to the input tensor.

    Implements the Stochastic Depth technique from "Deep Networks with Stochastic Depth"
    (https://arxiv.org/abs/1603.09382) used for randomly dropping residual branches
    of residual architectures.

    Args:
        input (torch.tensor): The input tensor of arbitrary dimensions with the first
            dimension being its batch size.
        p (float): Probability of the input to be zeroed.
        mode (str): Either "batch" or "row". "batch" randomly zeroes the entire input,
            "row" zeroes randomly selected rows from the batch.
        training (bool, optional): Apply stochastic depth if True. Defaults to True.

    Returns:
        torch.tensor: The randomly zeroed tensor.

    Raises:
        ValueError: If p is not between 0 and 1, or if mode is not "batch" or "row".

    Examples:
        >>> input_tensor = torch.randn(32, 64, 224, 224)
        >>> output = stochastic_depth(input_tensor, p=0.2, mode="batch")
        >>> print(output.shape)
        torch.Size([32, 64, 224, 224])

    Note:
        When training is False or p is 0, the function returns the input tensor unchanged.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "drop probability has to be between 0 and 1, but got {}".format(p)
        )
    if mode not in ["batch", "row"]:
        raise ValueError(
            "mode has to be either 'batch' or 'row', but got {}".format(mode)
        )
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate).div_(survival_rate)
    return input * noise


class StochasticDepth(nn.Module):
    """
    A module that applies stochastic depth to its input.

    This module implements the Stochastic Depth technique from the paper
    "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382).
    It randomly drops entire layers or parts of the input during training,
    which can help in regularizing deep networks.

    Attributes:
        p (float): The probability of dropping the input.
        mode (str): The mode of dropping, either "batch" or "row".

    Note:
        This implementation is a custom version and may need to be upgraded
        to use the official `torchvision.ops.StochasticDepth` in future versions.
    """

    def __init__(self, p: float, mode: str) -> None:
        # NOTE: need to upgrade to torchvision==0.11.0 to use StochasticDepth directly
        # from torchvision.ops import StochasticDepth
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        """
            Apply stochastic depth to the input tensor.

        This method applies the stochastic depth technique to the input tensor
        based on the probability and mode specified during initialization.

        Args:
            input (torch.Tensor): The input tensor to apply stochastic depth to.

        Returns:
            torch.Tensor: The output tensor after applying stochastic depth.

        Note:
            The behavior of this method depends on whether the module is in
            training mode or not, as well as the specified probability and mode.
        """
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "p=" + str(self.p)
        tmpstr += ", mode=" + str(self.mode)
        tmpstr += ")"
        return tmpstr


class DropoutNd(nn.Module):
    """
    A module that applies n-dimensional dropout to its input.

    This module extends the concept of dropout to n-dimensional tensors,
    allowing for more flexible dropout patterns in complex neural network
    architectures.

    Attributes:
        p (float): The probability of an element to be zeroed. Must be between 0 and 1.
        tie (bool): If True, ties the dropout mask across sequence lengths.
        transposed (bool): If True, assumes the input tensor is in a transposed format.

    Note:
        This implementation uses a custom dropout mechanism and may behave
        differently from PyTorch's built-in dropout functions.
    """

    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """Initialize dropout module.

        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p)
            )
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """
            Apply n-dimensional dropout to the input tensor.

        This method applies dropout to the input tensor based on the probability
        and configuration specified during initialization.

        Args:
            X (torch.Tensor): The input tensor of shape (batch, dim, lengths...).

        Returns:
            torch.Tensor: The output tensor after applying dropout.

        Note:
            The dropout is only applied during training mode. In evaluation mode,
            the input tensor is returned unchanged.

        Examples:
            >>> dropout = DropoutNd(p=0.5, tie=True, transposed=False)
            >>> input = torch.randn(32, 64, 10, 10)
            >>> output = dropout(input)
            >>> print(output.shape)
            torch.Size([32, 64, 10, 10])
        """
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
            # binomial = torch.distributions.binomial.Binomial(
            #   probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            return X
        return X


def Activation(activation=None, size=None, dim=-1):
    """
    Create and return an activation layer based on the specified activation type.

    This function provides a convenient way to create various activation layers
    commonly used in neural networks.

    Args:
        activation (str, optional): The type of activation to use. Supported values are:
            None, "id", "identity", "linear", "tanh", "relu", "gelu", "swish", "silu",
            "glu", "sigmoid", "sqrelu", "ln". Defaults to None.
        size (int, optional): The size parameter for certain activation types.
            Only used for specific activations. Defaults to None.
        dim (int, optional): The dimension along which to apply the activation for
            dimension-specific activations like GLU. Defaults to -1.

    Returns:
        nn.Module: An instance of the specified activation layer.

    Raises:
        NotImplementedError: If the specified activation is not implemented.

    Examples:
        >>> relu_activation = Activation("relu")
        >>> output = relu_activation(torch.randn(10, 20))
        >>> print(type(output))
        <class 'torch.Tensor'>

        >>> glu_activation = Activation("glu", dim=1)
        >>> output = glu_activation(torch.randn(10, 20, 30))
        >>> print(output.shape)
        torch.Size([10, 10, 30])

    Note:
        Some activation types (e.g., "ln") may require additional parameters
        that are not directly exposed in this function's interface.
    """
    if activation in [None, "id", "identity", "linear"]:
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation in ["swish", "silu"]:
        return nn.SiLU()
    elif activation == "glu":
        return nn.GLU(dim=dim)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "sqrelu":
        return SquaredReLU()
    elif activation == "ln":
        return TransposedLN(dim)
    else:
        raise NotImplementedError(
            "hidden activation '{}' is not implemented".format(activation)
        )


def get_initializer(name, activation=None):
    """
    Get an initialization function based on the specified name and activation.

    This function returns an initializer function that can be used to initialize
    the weights of neural network layers.

    Args:
        name (str): The name of the initializer. Supported values are:
            "uniform", "normal", "xavier", "zero", "one".
        activation (str, optional): The activation function used in the layer.
            This is used to determine the appropriate gain for certain initializers.
            Supported values are: None, "id", "identity", "linear", "modrelu", "relu",
            "tanh", "sigmoid", "gelu", "swish", "silu". Defaults to None.

    Returns:
        callable: An initialization function that can be applied to tensor weights.

    Raises:
        NotImplementedError: If the specified name or activation is not supported.

    Examples:
        >>> init_func = get_initializer("uniform", activation="relu")
        >>> linear = nn.Linear(10, 20)
        >>> init_func(linear.weight)

        >>> xavier_init = get_initializer("xavier")
        >>> conv = nn.Conv2d(3, 64, 3)
        >>> xavier_init(conv.weight)

    Note:
        The returned initializer functions are partial functions from PyTorch's
        initialization methods, with pre-set parameters based on the input arguments.
    """
    if activation in [None, "id", "identity", "linear", "modrelu"]:
        nonlinearity = "linear"
    elif activation in ["relu", "tanh", "sigmoid"]:
        nonlinearity = activation
    elif activation in ["gelu", "swish", "silu"]:
        nonlinearity = "relu"  # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(
            f"get_initializer: activation {activation} not supported"
        )

    if name == "uniform":
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == "normal":
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == "xavier":
        initializer = torch.nn.init.xavier_normal_
    elif name == "zero":
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == "one":
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(
            f"get_initializer: initializer type {name} not supported"
        )

    return initializer


def LinearActivation(
    d_input,
    d_output,
    bias=True,
    zero_bias_init=False,
    transposed=False,
    initializer=None,
    activation=None,
    activate=False,  # Apply activation as part of this module
    weight_norm=False,
    **kwargs,
):
    """
    Create a linear module with optional initialization and activation.

    This function constructs a linear layer with various options for initialization,
    activation, and normalization.

    Args:
        d_input (int): Dimension of the input features.
        d_output (int): Dimension of the output features.
        bias (bool, optional): If True, adds a learnable bias to the layer. Defaults to True.
        zero_bias_init (bool, optional): If True, initializes the bias to zero. Defaults to False.
        transposed (bool, optional): If True, uses TransposedLinear instead of nn.Linear. Defaults to False.
        initializer (str, optional): Initialization method for the weights. Defaults to None.
        activation (str, optional): Activation function to use. Defaults to None.
        activate (bool, optional): If True, applies the activation as part of this module. Defaults to False.
        weight_norm (bool, optional): If True, applies weight normalization. Defaults to False.
        **kwargs: Additional keyword arguments passed to the linear layer.

    Returns:
        nn.Module: A linear module, possibly followed by an activation function.

    Examples:
        >>> layer = LinearActivation(100, 200, activation='relu', activate=True)
        >>> input = torch.randn(32, 100)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([32, 200])

        >>> transposed_layer = LinearActivation(50, 75, transposed=True, weight_norm=True)
        >>> input = torch.randn(16, 50, 10)
        >>> output = transposed_layer(input)
        >>> print(output.shape)
        torch.Size([16, 75, 10])

    Note:
        If activation is set to 'glu', the output dimension is doubled internally
        before applying the activation.
    """
    # Construct core module
    # linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == "glu":
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output, dim=1 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class SquaredReLU(nn.Module):
    """
    A module that applies a squared ReLU activation function.

    This activation function first applies a standard ReLU (Rectified Linear Unit)
    activation, then squares the result. It can be used as a non-linear activation
    in neural networks, potentially capturing more complex patterns than standard ReLU.

    The function is defined as:
    f(x) = (max(0, x))^2

    Note:
        This activation function may lead to different gradient behavior compared
        to standard ReLU, potentially affecting the training dynamics of the network.
    """

    def forward(self, x):
        """
            Apply the squared ReLU activation function to the input.

        This method applies the ReLU function to the input and then squares the result.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the squared ReLU activation.

        Examples:
            >>> squared_relu = SquaredReLU()
            >>> input = torch.randn(10)
            >>> output = squared_relu(input)
            >>> print(output.shape)
            torch.Size([10])

        Note:
            For negative input values, the output will be zero. For positive input values,
            the output will be the square of the input.
        """
        return F.relu(x) ** 2


class TransposedLinear(nn.Module):
    """
    A linear module that operates on the second-to-last dimension of the input.

    This module implements a linear transformation similar to nn.Linear, but applies
    the transformation on the second-to-last dimension of the input tensor. It is
    designed to handle input tensors of shape (B, D, L), where B is the batch size,
    D is the input dimension, and L represents one or more additional dimensions.

    Attributes:
        weight (nn.Parameter): The learnable weights of the module of shape (d_output, d_input).
        bias (nn.Parameter or float): The learnable bias of the module of shape (d_output,).
            If bias is False, then the layer does not use a bias.

    Note:
        This module can be particularly useful in scenarios where the standard linear
        layer's dimension handling is not suitable, such as in certain types of
        sequence processing or when working with multi-dimensional data.
    """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # nn.Linear default init
        # nn.init.kaiming_uniform_(
        #   self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
            setattr(self.bias, "_optim", {"weight_decay": 0.0})
        else:
            self.bias = 0.0

    def forward(self, x):
        """
            Apply a linear transformation to the input tensor.

        This method performs the linear transformation on the second-to-last dimension
        of the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (B, D, ...), where B is the batch size,
                D is the input dimension, and ... represents any number of additional dimensions.

        Returns:
            torch.Tensor: The output tensor of shape (B, V, ...), where V is the output dimension.

        Examples:
            >>> transposed_linear = TransposedLinear(64, 128)
            >>> input = torch.randn(32, 64, 10, 10)
            >>> output = transposed_linear(input)
            >>> print(output.shape)
            torch.Size([32, 128, 10, 10])

        Note:
            The transformation is applied using einsum for efficient computation,
            especially for inputs with multiple trailing dimensions.
        """
        num_axis = len(x.shape[2:])  # num_axis in L, for broadcasting bias
        y = contract("b u ..., v u -> b v ...", x, self.weight) + self.bias.view(
            -1, *[1] * num_axis
        )
        return y


class TransposedLN(nn.Module):
    """
    A transposed Layer Normalization module.

    This module applies Layer Normalization over the second dimension of the input tensor.
    It is designed to handle input tensors of shape (B, D, L), where B is the batch size,
    D is the feature dimension to be normalized, and L represents one or more additional dimensions.

    Attributes:
        scalar (bool): If True, uses learnable scalar parameters for affine transformation.
            If False, uses a full LayerNorm.
        m (nn.Parameter): Learnable shift parameter when scalar is True.
        s (nn.Parameter): Learnable scale parameter when scalar is True.
        ln (nn.LayerNorm): LayerNorm module used when scalar is False.

    Note:
        This implementation may be slower than a dedicated CUDA/Triton implementation.
        Future optimizations could provide substantial end-to-end speedup.
    """

    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        """
            Apply transposed Layer Normalization to the input tensor.

        This method normalizes the input tensor along the second dimension (D).

        Args:
            x (torch.Tensor): The input tensor of shape (B, D, ...), where B is the batch size,
                D is the feature dimension to be normalized, and ... represents any number of
                additional dimensions.

        Returns:
            torch.Tensor: The normalized output tensor of the same shape as the input.

        Examples:
            >>> transposed_ln = TransposedLN(64, scalar=True)
            >>> input = torch.randn(32, 64, 10, 10)
            >>> output = transposed_ln(input)
            >>> print(output.shape)
            torch.Size([32, 64, 10, 10])

        Note:
            When scalar is True, it uses learnable scalar parameters for affine transformation.
            When scalar is False, it applies full LayerNorm by rearranging the tensor dimensions.
        """
        if self.scalar:
            # calc. stats over D dim / channels
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s / s) * (x - m + self.m)
        else:
            # move channel to last axis, apply layer_norm,
            # then move channel back to second axis
            _x = self.ln(rearrange(x, "b d ... -> b ... d"))
            y = rearrange(_x, "b ... d -> b d ...")
        return y


class Normalization(nn.Module):
    """
    A flexible normalization module supporting various normalization techniques.

    This module provides a unified interface for different types of normalization,
    including Layer Normalization, Instance Normalization, Batch Normalization,
    and Group Normalization. It can handle both standard and transposed input formats.

    Attributes:
        transposed (bool): If True, assumes the length dimension is -1 or -2.
        _name_ (str): The type of normalization to use. Options are "layer", "instance",
                      "batch", "group", or "none".
        channel (bool): If True, normalization is applied over the channel dimension.
        norm (nn.Module): The actual normalization module based on the specified type.

    Note:
        The behavior and performance of this module can vary significantly based on
        the chosen normalization type and the structure of the input data.
    """

    def __init__(
        self,
        d,
        transposed=False,  # Length dimension is -1 or -2
        _name_="layer",
        **kwargs,
    ):
        super().__init__()
        self.transposed = transposed
        self._name_ = _name_

        if _name_ == "layer":
            self.channel = True  # Normalize over channel dimension
            if self.transposed:
                self.norm = TransposedLN(d, **kwargs)
            else:
                self.norm = nn.LayerNorm(d, **kwargs)
        elif _name_ == "instance":
            self.channel = False
            norm_args = {"affine": False, "track_running_stats": False}
            norm_args.update(kwargs)
            self.norm = nn.InstanceNorm1d(
                d, **norm_args
            )  # (True, True) performs very poorly
        elif _name_ == "batch":
            self.channel = False
            norm_args = {"affine": True, "track_running_stats": True}
            norm_args.update(kwargs)
            self.norm = nn.BatchNorm1d(d, **norm_args)
        elif _name_ == "group":
            self.channel = False
            self.norm = nn.GroupNorm(1, d, *kwargs)
        elif _name_ == "none":
            self.channel = True
            self.norm = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        """
            Apply the specified normalization to the input tensor.

        This method reshapes the input tensor if necessary, applies the chosen
        normalization technique, and then restores the original shape.

        Args:
            x (torch.Tensor): The input tensor to be normalized. The shape depends on
                the 'transposed' attribute and the specific normalization technique.

        Returns:
            torch.Tensor: The normalized tensor, maintaining the same shape as the input.

        Examples:
            >>> norm = Normalization(64, transposed=False, _name_="layer")
            >>> input = torch.randn(32, 10, 64)
            >>> output = norm(input)
            >>> print(output.shape)
            torch.Size([32, 10, 64])

        Note:
            The method handles higher dimensional inputs by reshaping them before
            normalization and then restoring the original shape afterwards.
        """
        # Handle higher dimension logic
        shape = x.shape
        if self.transposed:
            x = rearrange(x, "b d ... -> b d (...)")
        else:
            x = rearrange(x, "b ... d -> b (...)d ")

        # The cases of LayerNorm / no normalization
        # are automatically handled in all cases
        # Instance/Batch Norm work automatically with transposed axes
        if self.channel or self.transposed:
            x = self.norm(x)
        else:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)

        x = x.view(shape)
        return x

    def step(self, x, **kwargs):
        """
            Apply normalization to a single step input.

        This method is designed for use in scenarios where normalization needs to be
        applied to a single time step or element, such as in recurrent neural networks
        or sequential processing.

        Args:
            x (torch.Tensor): The input tensor representing a single step or element.
            **kwargs: Additional keyword arguments that might be required for specific
                      normalization types.

        Returns:
            torch.Tensor: The normalized tensor for the single step.

        Raises:
            AssertionError: If the normalization type is not one of "layer", "instance",
                            "batch", "group", or "none".

        Examples:
            >>> norm = Normalization(64, transposed=True, _name_="layer")
            >>> step_input = torch.randn(32, 64)
            >>> normalized_step = norm.step(step_input)
            >>> print(normalized_step.shape)
            torch.Size([32, 64])

        Note:
            This method adds an extra dimension to the input if 'transposed' is True,
            applies normalization, and then squeezes the added dimension.
        """
        assert self._name_ in ["layer", "instance", "batch", "group", "none"]
        if self.transposed:
            x = x.unsqueeze(-1)
        x = self.forward(x)
        if self.transposed:
            x = x.squeeze(-1)
        return x


class TSNormalization(nn.Module):
    """
    A module for time series normalization.

    This class implements normalization techniques specifically designed for time series data.
    It supports different methods of normalization based on the temporal characteristics of the input.

    Attributes:
        method (str): The normalization method to use. Supported values are "mean" and "last".
        horizon (int): The number of time steps to consider for normalization.

    Note:
        This normalization is particularly useful in time series forecasting tasks,
        where different parts of the time series may need to be treated differently
        for effective normalization.
    """

    def __init__(self, method, horizon):
        super().__init__()

        self.method = method
        self.horizon = horizon

    def forward(self, x):
        """
            Apply time series normalization to the input tensor.

        This method normalizes the input time series data based on the specified method and horizon.

        Args:
            x (torch.Tensor): The input tensor of shape (B, L, D), where B is the batch size,
                L is the sequence length, and D is the feature dimension.

        Returns:
            torch.Tensor: The normalized tensor of the same shape as the input.

        Examples:
            >>> ts_norm = TSNormalization(method="mean", horizon=24)
            >>> input = torch.randn(32, 100, 64)  # 32 batches, 100 time steps, 64 features
            >>> output = ts_norm(input)
            >>> print(output.shape)
            torch.Size([32, 100, 64])

        Note:
            - For the "mean" method, it uses the mean of absolute values up to the horizon for normalization.
            - For the "last" method, it uses the last value before the horizon for normalization.
            - If the method is neither "mean" nor "last", the input is returned unchanged.
        """
        # x must be BLD
        if self.method == "mean":
            self.scale = x.abs()[:, : -self.horizon].mean(dim=1)[:, None, :]
            return x / self.scale
        elif self.method == "last":
            self.scale = x.abs()[:, -self.horizon - 1][:, None, :]
            return x / self.scale
        return x


class TSInverseNormalization(nn.Module):
    """
    A module for inverting time series normalization.

    This class implements the inverse operation of TSNormalization, allowing
    the restoration of normalized time series data to its original scale.

    Attributes:
        method (str): The normalization method that was used. Supported values are "mean" and "last".
        normalizer (TSNormalization): The TSNormalization instance that was used for the initial normalization.

    Note:
        This module is typically used in conjunction with TSNormalization to revert
        normalized predictions or processed data back to their original scale in
        time series analysis and forecasting tasks.
    """

    def __init__(self, method, normalizer):
        super().__init__()

        self.method = method
        self.normalizer = normalizer

    def forward(self, x):
        """
            Apply inverse time series normalization to the input tensor.

        This method reverses the normalization applied by TSNormalization, restoring
        the data to its original scale.

        Args:
            x (torch.Tensor): The normalized input tensor of shape (B, L, D), where B is the batch size,
                L is the sequence length, and D is the feature dimension.

        Returns:
            torch.Tensor: The denormalized tensor of the same shape as the input.

        Examples:
            >>> ts_norm = TSNormalization(method="mean", horizon=24)
            >>> ts_inverse_norm = TSInverseNormalization(method="mean", normalizer=ts_norm)
            >>> original_input = torch.randn(32, 100, 64)
            >>> normalized = ts_norm(original_input)
            >>> denormalized = ts_inverse_norm(normalized)
            >>> print(torch.allclose(original_input, denormalized))
            True

        Note:
            - For the "mean" and "last" methods, it multiplies the input by the scale stored in the normalizer.
            - If the method is neither "mean" nor "last", the input is returned unchanged.
        """
        if self.method == "mean" or self.method == "last":
            return x * self.normalizer.scale
        return x


class ReversibleInstanceNorm1dInput(nn.Module):
    """
    A reversible instance normalization module for 1D input.

    This module applies instance normalization to the input in a way that allows
    for reversing the normalization process later. It is designed to work with
    1D data, such as time series or sequential data.

    Attributes:
        transposed (bool): If True, expects input in BDL format, otherwise in BLD format.
        norm (nn.InstanceNorm1d): The instance normalization layer.
        s (torch.Tensor): Standard deviation of the input, computed during forward pass.
        m (torch.Tensor): Mean of the input, computed during forward pass.

    Note:
        This module is typically used in pairs with ReversibleInstanceNorm1dOutput
        to allow for normalization that can be undone, which can be useful in
        certain types of neural network architectures or processing pipelines.
    """

    def __init__(self, d, transposed=False):
        super().__init__()
        # BLD if transpoed is False, otherwise BDL
        self.transposed = transposed
        self.norm = nn.InstanceNorm1d(d, affine=True, track_running_stats=False)

    def forward(self, x):
        """
            Apply reversible instance normalization to the input tensor.

        This method normalizes the input tensor and stores the statistics needed for
        reversing the normalization later.

        Args:
            x (torch.Tensor): The input tensor. If transposed is False, expected shape is (B, L, D),
                otherwise (B, D, L), where B is batch size, L is sequence length, and D is feature dimension.

        Returns:
            torch.Tensor: The normalized tensor with the same shape as the input.

        Examples:
            >>> rev_norm = ReversibleInstanceNorm1dInput(64, transposed=False)
            >>> input = torch.randn(32, 100, 64)  # 32 batches, 100 time steps, 64 features
            >>> output = rev_norm(input)
            >>> print(output.shape)
            torch.Size([32, 100, 64])

        Note:
            - The method computes and stores the mean and standard deviation of the input.
            - A small epsilon (1e-4) is added to the standard deviation to avoid division by zero.
            - The normalization is applied along the last dimension for non-transposed input,
              and along the second-to-last dimension for transposed input.
        """
        # Means, stds
        if not self.transposed:
            x = x.transpose(-1, -2)

        self.s, self.m = torch.std_mean(x, dim=-1, unbiased=False, keepdim=True)
        self.s += 1e-4

        x = (x - self.m) / self.s
        # x = self.norm.weight.unsqueeze(-1) * x + self.norm.bias.unsqueeze(-1)

        if not self.transposed:
            return x.transpose(-1, -2)
        return x


class ReversibleInstanceNorm1dOutput(nn.Module):
    """
    A module for reversing the instance normalization applied by ReversibleInstanceNorm1dInput.

    This module is designed to work in conjunction with ReversibleInstanceNorm1dInput
    to undo the normalization process, restoring the data to its original scale and distribution.

    Attributes:
        transposed (bool): If True, expects input in BDL format, otherwise in BLD format.
        weight (nn.Parameter): The weight parameter from the input normalization module.
        bias (nn.Parameter): The bias parameter from the input normalization module.
        norm_input (ReversibleInstanceNorm1dInput): The input normalization module used for the forward pass.

    Note:
        This module should be used with tensors that have been normalized by a corresponding
        ReversibleInstanceNorm1dInput instance to ensure correct denormalization.
    """

    def __init__(self, norm_input):
        super().__init__()
        self.transposed = norm_input.transposed
        self.weight = norm_input.norm.weight
        self.bias = norm_input.norm.bias
        self.norm_input = norm_input

    def forward(self, x):
        """
            Reverse the instance normalization applied to the input tensor.

        This method denormalizes the input tensor using the statistics stored in the
        corresponding ReversibleInstanceNorm1dInput module.

        Args:
            x (torch.Tensor): The normalized input tensor. If transposed is False, expected shape is (B, L, D),
                otherwise (B, D, L), where B is batch size, L is sequence length, and D is feature dimension.

        Returns:
            torch.Tensor: The denormalized tensor with the same shape as the input.

        Examples:
            >>> rev_norm_input = ReversibleInstanceNorm1dInput(64, transposed=False)
            >>> rev_norm_output = ReversibleInstanceNorm1dOutput(rev_norm_input)
            >>> original = torch.randn(32, 100, 64)
            >>> normalized = rev_norm_input(original)
            >>> denormalized = rev_norm_output(normalized)
            >>> print(torch.allclose(original, denormalized, atol=1e-6))
            True

        Note:
            - The denormalization process uses the mean and standard deviation stored in the norm_input attribute.
            - If the input was transposed during normalization, it will be transposed again during denormalization.
        """
        if not self.transposed:
            x = x.transpose(-1, -2)

        # x = (x - self.bias.unsqueeze(-1))/self.weight.unsqueeze(-1)
        x = x * self.norm_input.s + self.norm_input.m

        if not self.transposed:
            return x.transpose(-1, -2)
        return x
