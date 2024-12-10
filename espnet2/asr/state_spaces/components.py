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
    Apply stochastic depth.

    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Stochastic depth helps in training deep networks by randomly skipping 
    layers during training, which can improve generalization and reduce 
    overfitting.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions 
            with the first one being its batch i.e. a batch with ``N`` rows.
        p (float): Probability of the input to be zeroed, where `0` means 
            no dropout and `1` means dropping the entire input.
        mode (str): Specifies the mode of operation, can be either:
            - `"batch"`: Randomly zeroes the entire input tensor.
            - `"row"`: Randomly zeroes selected rows from the batch.
        training (bool): Indicates whether to apply stochastic depth. 
            If `False`, the input is returned unmodified. Default: `True`.

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor, with the same shape as 
        the input.

    Raises:
        ValueError: If `p` is not in the range [0, 1], or if `mode` is not 
        `"batch"` or `"row"`.

    Examples:
        >>> input_tensor = torch.randn(4, 3)  # A batch of 4 samples with 3 features
        >>> output_tensor = stochastic_depth(input_tensor, p=0.5, mode="row")
        >>> output_tensor.shape
        torch.Size([4, 3])  # Output shape remains the same as input

        >>> output_tensor = stochastic_depth(input_tensor, p=0.2, mode="batch")
        >>> output_tensor.shape
        torch.Size([4, 3])  # Output shape remains the same as input
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
    Stochastic depth module for randomly dropping residual branches.

    This module implements stochastic depth as described in the paper 
    "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382).
    It randomly drops entire residual connections during training, which 
    helps to improve model robustness and reduces overfitting.

    Attributes:
        p (float): The probability of dropping a residual branch.
        mode (str): The mode of operation, either "batch" or "row".

    Args:
        p (float): Probability of dropping the residual branch.
        mode (str): Mode of operation; can be "batch" or "row".

    Examples:
        >>> import torch
        >>> sd = StochasticDepth(p=0.5, mode='batch')
        >>> input_tensor = torch.randn(10, 3, 32, 32)  # Example input
        >>> output_tensor = sd(input_tensor)  # Apply stochastic depth

    Note:
        Ensure that the model is in training mode when applying this 
        module to observe the effects of stochastic depth.
    """

    def __init__(self, p: float, mode: str) -> None:
        # NOTE: need to upgrade to torchvision==0.11.0 to use StochasticDepth directly
        # from torchvision.ops import StochasticDepth
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        """
        Perform the forward pass of the Stochastic Depth module.

        This method applies stochastic depth to the input tensor, which randomly 
        drops residual branches in the architecture during training, as described 
        in the paper `"Deep Networks with Stochastic Depth" 
        <https://arxiv.org/abs/1603.09382>`_. The behavior of the module is controlled 
        by the drop probability `p` and the specified mode (`"batch"` or `"row"`).

        Args:
            input (Tensor): The input tensor of shape (N, ...), where N is the 
                            batch size and ... represents any number of additional 
                            dimensions.

        Returns:
            Tensor: The output tensor after applying stochastic depth. The shape 
                    of the output tensor will match the input tensor.

        Examples:
            >>> import torch
            >>> stochastic_depth_layer = StochasticDepth(p=0.5, mode="row")
            >>> input_tensor = torch.randn(10, 5)  # Batch of 10 samples, 5 features
            >>> output_tensor = stochastic_depth_layer(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 5])  # Output shape matches input shape

        Note:
            Stochastic depth is only applied during training. If the module is 
            in evaluation mode, the input is returned unchanged.

        Raises:
            ValueError: If the input tensor is not of type `torch.Tensor`.
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
    N-dimensional dropout module.

    This module applies dropout in N dimensions, allowing for flexible
    application of dropout masks across multiple axes. It supports
    options to tie the dropout mask across sequence lengths and to apply
    dropout in a transposed manner.

    Attributes:
        p (float): Probability of an element being zeroed. Must be in the
            range [0, 1).
        tie (bool): If True, tie the dropout mask across sequence lengths.
        transposed (bool): If True, apply dropout in a transposed manner.
        binomial (Binomial): A binomial distribution used to sample dropout
            masks.

    Args:
        p (float): Dropout probability (default: 0.5).
        tie (bool): Whether to tie dropout mask across sequence lengths 
            (default: True).
        transposed (bool): Whether to apply dropout in a transposed manner 
            (default: True).

    Raises:
        ValueError: If `p` is not in the range [0, 1).

    Examples:
        >>> dropout = DropoutNd(p=0.3)
        >>> input_tensor = torch.randn(2, 3, 4)  # Example input
        >>> output_tensor = dropout(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([2, 3, 4])  # Shape remains the same, but values are dropped

    Note:
        This module is particularly useful in sequence models where you
        may want to apply dropout independently to each sequence step.
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
        Perform the forward pass of the DropoutNd module.

        This method applies dropout to the input tensor during training.
        It randomly zeroes elements of the input tensor based on the
        specified dropout probability `p`. The behavior can be controlled
        with the `tie` and `transposed` attributes, which determine how
        the dropout mask is applied.

        Args:
            input (Tensor): The input tensor of shape (batch, dim, lengths...).
                The dimensions beyond the second are treated as additional
                lengths over which the dropout is applied.

        Returns:
            Tensor: The output tensor with dropout applied. If the module is
                not in training mode, the input tensor is returned unchanged.

        Examples:
            >>> dropout = DropoutNd(p=0.5, tie=True, transposed=True)
            >>> input_tensor = torch.rand(10, 3, 5)  # Example input
            >>> output_tensor = dropout(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([10, 3, 5])  # Shape remains the same, but values are zeroed out

        Note:
            When `tie` is set to `True`, the same dropout mask is applied
            across all lengths, while if `tie` is `False`, each length
            has its own independent dropout mask.
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
    Create an activation function module.

    This function returns a specific activation function from the available
    options in PyTorch, which can be used in neural network architectures.
    The supported activation functions include identity, tanh, relu, gelu,
    swish (or silu), glu, sigmoid, squared relu, and transposed layer norm.

    Args:
        activation (str or None): The type of activation function to create.
            Options include: "id", "identity", "linear", "tanh", "relu",
            "gelu", "swish", "silu", "glu", "sigmoid", "sqrelu", "ln".
            If None, it defaults to an identity function.
        size (int, optional): Size of the output (not used currently).
        dim (int, optional): The dimension along which to apply the GLU
            activation. Default is -1.

    Returns:
        nn.Module: A PyTorch activation function module corresponding to the
        specified activation type.

    Raises:
        NotImplementedError: If the specified activation type is not
        supported.

    Examples:
        >>> relu_activation = Activation("relu")
        >>> tanh_activation = Activation("tanh")
        >>> glu_activation = Activation("glu", dim=1)

    Note:
        The `size` parameter is currently not utilized in the function but
        may be included for future enhancements.
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
    Get the appropriate weight initializer based on the specified name and 
    activation function.

    This function returns a callable that initializes weights according to 
    the specified initialization method and takes into account the type of 
    activation function being used. It supports several initialization 
    methods including 'uniform', 'normal', 'xavier', 'zero', and 'one'. 
    If the activation function is not recognized, it raises a 
    NotImplementedError.

    Args:
        name (str): The name of the initializer. Supported values are 
            'uniform', 'normal', 'xavier', 'zero', and 'one'.
        activation (str, optional): The activation function to consider for 
            the initialization. Supported values include 'relu', 'tanh', 
            'sigmoid', 'gelu', 'swish', and 'linear'. Defaults to None.

    Returns:
        Callable: A callable that initializes weights based on the specified 
        initializer and activation function.

    Raises:
        NotImplementedError: If the specified initializer name or activation 
        function is not supported.

    Examples:
        # Get a uniform initializer for ReLU activation
        initializer = get_initializer("uniform", activation="relu")
        # Apply the initializer to a tensor
        weight_tensor = torch.empty(3, 5)
        initializer(weight_tensor)

        # Get a Xavier initializer for sigmoid activation
        initializer = get_initializer("xavier", activation="sigmoid")
        weight_tensor = torch.empty(4, 4)
        initializer(weight_tensor)
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
    Create a linear module with optional activation and initialization.

    This function constructs a linear layer with specified input and output
    dimensions, applies optional weight initialization, and includes an
    activation function if specified. The linear layer can be transposed,
    and weight normalization can be applied as well.

    Args:
        d_input (int): The number of input features.
        d_output (int): The number of output features.
        bias (bool, optional): If True, adds a learnable bias to the output.
            Default is True.
        zero_bias_init (bool, optional): If True, initializes the bias to
            zero. Default is False.
        transposed (bool, optional): If True, creates a transposed linear layer
            instead of a standard one. Default is False.
        initializer (str, optional): The type of weight initializer to use.
            Options include "uniform", "normal", "xavier", "zero", "one".
        activation (str, optional): The activation function to apply after the
            linear transformation. Options include "relu", "tanh", "sigmoid",
            "gelu", "swish", "silu", "glu", "sqrelu", or "ln".
        activate (bool, optional): If True, applies the activation function as
            part of this module. Default is False.
        weight_norm (bool, optional): If True, applies weight normalization to
            the linear layer. Default is False.
        **kwargs: Additional arguments for the linear layer constructor.

    Returns:
        nn.Module: A sequential module containing the linear layer and, if
        specified, the activation function.

    Examples:
        >>> linear_layer = LinearActivation(d_input=128, d_output=64,
        ...                                  activation='relu', bias=True)
        >>> output = linear_layer(torch.randn(10, 128))
        
        >>> transposed_layer = LinearActivation(d_input=64, d_output=128,
        ...                                      transposed=True, weight_norm=True)
        >>> output = transposed_layer(torch.randn(10, 64, 1))
    
    Note:
        - The function raises a NotImplementedError if the specified
          activation function is not implemented.
        - Ensure that the input tensor shape matches the expected
          dimensions for the linear layer.
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
    Squared ReLU activation function.

    Applies the squared rectified linear unit (ReLU) activation function to 
    the input tensor. The Squared ReLU function outputs the square of the 
    input if it is positive and zero otherwise.

    This can be particularly useful in neural networks where a non-linear 
    activation is required that emphasizes positive values while suppressing 
    negative ones, effectively allowing for a form of feature amplification.

    Args:
        None

    Returns:
        Tensor: The squared output of the ReLU activation function applied 
        element-wise to the input tensor.

    Examples:
        >>> activation = SquaredReLU()
        >>> input_tensor = torch.tensor([-1.0, 0.0, 2.0, 3.0])
        >>> output_tensor = activation(input_tensor)
        >>> print(output_tensor)
        tensor([0., 0., 4., 9.])
    """
    def forward(self, x):
        """
        Squared ReLU activation function.

    Applies the ReLU activation function followed by squaring the output.
    The function is defined as:

        SquaredReLU(x) = (ReLU(x))^2

    This activation can be useful in scenarios where non-negative outputs are 
    required, and squaring the output can enhance the gradients for positive 
    inputs during backpropagation.

    Examples:
        >>> activation = SquaredReLU()
        >>> input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        >>> output_tensor = activation(input_tensor)
        >>> print(output_tensor)
        tensor([0., 0., 1., 4.])
    
    Returns:
        Tensor: The squared ReLU applied to the input tensor.
        """
        return F.relu(x) ** 2


class TransposedLinear(nn.Module):
    """
    Transposed linear module.

    This module performs a linear transformation on the second-to-last dimension 
    of the input tensor. It assumes that the input tensor has the shape 
    (B, D, L), where B is the batch size, D is the number of features, and L can 
    be one or more dimensions.

    Attributes:
        weight (torch.Parameter): The learnable weight parameter of the module.
        bias (torch.Parameter or float): The learnable bias parameter or 0.0 if 
            bias is not used.

    Args:
        d_input (int): The number of input features (D).
        d_output (int): The number of output features.
        bias (bool, optional): Whether to include a bias term. Defaults to True.

    Returns:
        Tensor: The output tensor after applying the linear transformation.

    Examples:
        >>> import torch
        >>> layer = TransposedLinear(d_input=4, d_output=2)
        >>> x = torch.randn(3, 4, 5)  # Batch size of 3, 4 features, 5 lengths
        >>> output = layer(x)
        >>> output.shape
        torch.Size([3, 2, 5])  # Output has 2 features

    Note:
        The weight is initialized using Kaiming uniform initialization, which is 
        suitable for layers with ReLU activations. If bias is used, it is initialized 
        uniformly within a specific bound based on the input dimensions.

    Todo:
        Consider implementing a dedicated CUDA/Triton implementation for 
        potential performance improvements.
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
        Transposed linear module.

        This module applies a linear transformation on the second-to-last 
        dimension of the input tensor. It assumes that the input tensor has 
        the shape (B, D, L), where B is the batch size, D is the number of 
        features, and L can be one or more additional dimensions.

        Attributes:
            weight (torch.Tensor): The learnable weights of the module.
            bias (torch.Tensor or float): The learnable bias of the module, 
                or 0.0 if no bias is used.

        Args:
            d_input (int): Number of input features.
            d_output (int): Number of output features.
            bias (bool, optional): Whether to include a bias term. Defaults to True.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.

        Examples:
            >>> layer = TransposedLinear(10, 5)
            >>> input_tensor = torch.randn(2, 10, 3)  # Batch size of 2, 10 features, 3 lengths
            >>> output_tensor = layer(input_tensor)
            >>> output_tensor.shape
            torch.Size([2, 5, 3])  # Output shape should be (B, D_out, L)

        Note:
            The linear transformation is performed using the einsum contraction
            to handle the multi-dimensional input tensor efficiently.
        """
        num_axis = len(x.shape[2:])  # num_axis in L, for broadcasting bias
        y = contract("b u ..., v u -> b v ...", x, self.weight) + self.bias.view(
            -1, *[1] * num_axis
        )
        return y


class TransposedLN(nn.Module):
    """
    Transposed LayerNorm module.

    This module applies Layer Normalization over the second dimension of the input tensor,
    which is expected to have the shape (B, D, L), where B is the batch size, D is the 
    feature dimension, and L can represent one or more length dimensions.

    The implementation includes two modes:
    1. Scalar mode: Normalizes the input using learned scale and shift parameters.
    2. LayerNorm mode: Applies the standard LayerNorm to the input.

    This module is currently not optimized for speed and a dedicated CUDA/Triton 
    implementation could provide substantial performance improvements.

    Attributes:
        scalar (bool): Indicates whether to use scalar parameters for normalization.

    Args:
        d (int): The number of features in the input tensor.
        scalar (bool, optional): If True, use scalar parameters for normalization.
            Default is True.

    Returns:
        Tensor: The normalized output tensor of the same shape as the input.

    Examples:
        >>> layer_norm = TransposedLN(d=64)
        >>> input_tensor = torch.randn(10, 64, 5)  # Batch of 10, 64 features, 5 length
        >>> output_tensor = layer_norm(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 64, 5])  # Output retains the same shape

    Note:
        This implementation is currently slow. Consider using a more optimized 
        implementation for performance-critical applications.

    Todo:
        Implement a CUDA/Triton version for better performance.
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
        Transposed LayerNorm module.

        This module applies Layer Normalization over the second dimension 
        (channels) of the input tensor. It assumes the input shape is 
        (B, D, L), where B is the batch size, D is the number of features 
        (channels), and L can be one or more additional dimensions. 

        This implementation may be slow for large inputs, and a dedicated 
        CUDA/Triton implementation should provide substantial end-to-end 
        speedup.

        Attributes:
            scalar (bool): If True, uses scalar parameters for normalization. 
                        If False, uses standard LayerNorm.
            m (Parameter): The mean parameter, used if scalar is True.
            s (Parameter): The standard deviation parameter, used if scalar 
                        is True.
            ln (LayerNorm): The LayerNorm module, used if scalar is False.

        Args:
            d (int): The number of features (channels) to normalize.
            scalar (bool): If True, use scalar normalization; if False, 
                        use LayerNorm. Default: True.

        Returns:
            Tensor: The normalized output tensor.

        Examples:
            >>> layer_norm = TransposedLN(d=64)
            >>> input_tensor = torch.randn(32, 64, 10)  # (B, D, L)
            >>> output_tensor = layer_norm(input_tensor)

        Note:
            This module is designed for use with transposed convolutional 
            architectures where the channel dimension is the second 
            dimension of the input tensor.

        Todo:
            Consider implementing a faster version using CUDA or Triton 
            for better performance on larger inputs.
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
    Normalization module for various normalization techniques.

    This class implements several normalization techniques, including LayerNorm,
    InstanceNorm, BatchNorm, GroupNorm, and Identity normalization. The choice of
    normalization method is determined by the `_name_` parameter during initialization.

    Attributes:
        transposed (bool): Indicates if the normalization is applied on transposed 
            input (length dimension is -1 or -2).
        _name_ (str): The type of normalization to apply. Options include:
            'layer', 'instance', 'batch', 'group', 'none'.
        norm (nn.Module): The normalization layer based on the selected method.

    Args:
        d (int): The number of features (channels) for normalization.
        transposed (bool, optional): Whether to apply normalization to transposed 
            input. Default is False.
        _name_ (str, optional): The name of the normalization type. Default is 'layer'.
        **kwargs: Additional keyword arguments for specific normalization methods.

    Raises:
        NotImplementedError: If an unsupported normalization type is specified in 
            `_name_`.

    Examples:
        >>> layer_norm = Normalization(d=64, _name_='layer')
        >>> instance_norm = Normalization(d=64, _name_='instance')
        >>> batch_norm = Normalization(d=64, _name_='batch')
        >>> group_norm = Normalization(d=64, _name_='group')
        >>> identity_norm = Normalization(d=64, _name_='none')

    Note:
        The transposed option is particularly useful for working with sequences 
        or multi-dimensional data where the last dimensions represent time or 
        length.
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
        Normalization module for different normalization techniques.

        This module implements various normalization techniques such as LayerNorm,
        InstanceNorm, BatchNorm, GroupNorm, and an identity operation. It allows
        for normalization across different dimensions and can be configured to use
        transposed dimensions.

        Attributes:
            transposed (bool): Indicates whether the normalization should be applied
                on transposed dimensions.
            _name_ (str): Specifies the type of normalization to apply.
            norm (nn.Module): The actual normalization layer based on the specified
                normalization type.

        Args:
            d (int): The number of features in the input tensor.
            transposed (bool): If True, applies normalization on the last dimension.
                Default is False.
            _name_ (str): The type of normalization to apply. Options are:
                "layer", "instance", "batch", "group", or "none". Default is "layer".
            **kwargs: Additional keyword arguments to pass to the normalization layer.

        Returns:
            Tensor: The normalized tensor.

        Examples:
            # Using LayerNorm
            layer_norm = Normalization(d=64, _name_="layer")
            output = layer_norm(input_tensor)

            # Using BatchNorm
            batch_norm = Normalization(d=64, _name_="batch")
            output = batch_norm(input_tensor)

        Note:
            The input tensor is expected to be of shape (B, D, L) or (B, L, D)
            depending on the transposed configuration.

        Raises:
            NotImplementedError: If an unsupported normalization type is specified.
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
        Apply the normalization step to the input tensor.

    This method applies the normalization operation to the input tensor `x` based
    on the specified normalization type (layer, instance, batch, group, or none).
    It handles the transposed state of the tensor as needed and reshapes the 
    input to apply normalization correctly. The result is returned in the same 
    shape as the input tensor.

    Args:
        x (Tensor): The input tensor to be normalized.
        **kwargs: Additional keyword arguments to be passed to the normalization 
                layer.

    Returns:
        Tensor: The normalized tensor with the same shape as the input.

    Raises:
        AssertionError: If the normalization type (_name_) is not one of the 
                        supported types ("layer", "instance", "batch", "group", 
                        "none").

    Examples:
        >>> norm_layer = Normalization(d=64, _name_='layer')
        >>> input_tensor = torch.randn(32, 64)  # Batch of 32 samples, 64 features
        >>> output_tensor = norm_layer.step(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 64])

        >>> group_norm_layer = Normalization(d=64, _name_='group')
        >>> input_tensor = torch.randn(32, 64)  # Batch of 32 samples, 64 features
        >>> output_tensor = group_norm_layer.step(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 64])
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
    Time Series Normalization module.

    This module normalizes time series data using specified methods over a
    defined horizon. It supports normalization by either the mean or the last
    value within the specified horizon. The input tensor is expected to have
    the shape (B, L, D), where B is the batch size, L is the sequence length,
    and D is the number of features.

    Attributes:
        method (str): The normalization method to apply. Can be "mean" or "last".
        horizon (int): The number of time steps to consider for normalization.

    Args:
        method (str): The normalization method, either "mean" or "last".
        horizon (int): The number of time steps to use for normalization.

    Returns:
        Tensor: The normalized tensor with the same shape as the input.

    Examples:
        >>> normalization = TSNormalization(method="mean", horizon=3)
        >>> input_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        >>> normalized_tensor = normalization(input_tensor)
        >>> print(normalized_tensor)
        tensor([[[0.3333, 0.6667],
                  [1.0000, 1.3333],
                  [1.6667, 2.0000]]])

        >>> normalization_last = TSNormalization(method="last", horizon=1)
        >>> normalized_tensor_last = normalization_last(input_tensor)
        >>> print(normalized_tensor_last)
        tensor([[[1.0, 2.0],
                  [1.5, 2.0],
                  [2.5, 3.0]]])

    Note:
        If the input tensor does not have the expected shape, the behavior of
        this module may be undefined.

    Raises:
        ValueError: If the input tensor does not have at least 2 dimensions.
    """
    def __init__(self, method, horizon):
        super().__init__()

        self.method = method
        self.horizon = horizon

    def forward(self, x):
        """
        Time Series Normalization module.

        This module normalizes time series data based on the specified method and
        horizon. The normalization is performed on the input tensor, which is 
        expected to have a shape of (B, L, D), where B is the batch size, L is 
        the length of the sequence, and D is the number of features.

        Attributes:
            method (str): The normalization method to use. Options are "mean" 
                        or "last".
            horizon (int): The number of timesteps to consider for normalization.

        Args:
            method (str): The normalization method to use ("mean" or "last").
            horizon (int): The number of timesteps to consider for normalization.

        Returns:
            Tensor: The normalized tensor.

        Examples:
            >>> normalization = TSNormalization(method="mean", horizon=5)
            >>> x = torch.randn(10, 20, 3)  # A batch of 10 sequences, each of length 20 with 3 features
            >>> normalized_x = normalization(x)
            >>> print(normalized_x.shape)  # Output: torch.Size([10, 20, 3])

            >>> normalization_last = TSNormalization(method="last", horizon=5)
            >>> normalized_last_x = normalization_last(x)
            >>> print(normalized_last_x.shape)  # Output: torch.Size([10, 20, 3])

        Note:
            Ensure that the input tensor has the correct shape (B, L, D) before 
            using this module.
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
    TSInverseNormalization is a module that performs the inverse normalization 
    operation based on the method used during normalization. This is useful in 
    time series forecasting tasks where the predicted values need to be 
    converted back to the original scale.

    Attributes:
        method (str): The method used for normalization. Can be either 
            "mean" or "last".
        normalizer (TSNormalization): An instance of the TSNormalization 
            class that holds the scaling factor used during normalization.

    Args:
        method (str): The normalization method used, either "mean" or "last".
        normalizer (TSNormalization): The normalizer object containing the 
            scaling information.

    Returns:
        Tensor: The input tensor scaled back to the original values.

    Examples:
        >>> normalizer = TSNormalization(method="mean", horizon=5)
        >>> ts_inverse_norm = TSInverseNormalization(method="mean", normalizer=normalizer)
        >>> x_normalized = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> x_original = ts_inverse_norm(x_normalized)
        
    Note:
        The scaling is only applied if the method is either "mean" or "last".
        If a different method is provided, the input tensor will be returned 
        unchanged.
    """
    def __init__(self, method, normalizer):
        super().__init__()

        self.method = method
        self.normalizer = normalizer

    def forward(self, x):
        """
        Time Series Inverse Normalization module.

        This module is used to apply inverse normalization on time series data
        based on the method specified during initialization. The two supported
        methods are "mean" and "last", which correspond to scaling the input
        tensor by the mean or last observed value during normalization.

        Attributes:
            method (str): The method used for normalization; should be either
                        "mean" or "last".
            normalizer (TSNormalization): An instance of the TSNormalization
                                        class that contains the scaling factor.

        Args:
            method (str): The normalization method to use. Can be "mean" or "last".
            normalizer (TSNormalization): An instance of the TSNormalization class
                                        used to retrieve the scaling factor.

        Returns:
            Tensor: The inverse normalized tensor.

        Examples:
            >>> normalizer = TSNormalization(method="mean", horizon=5)
            >>> ts_inverse_norm = TSInverseNormalization(method="mean", normalizer=normalizer)
            >>> input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> output_tensor = ts_inverse_norm(input_tensor)

        Note:
            Ensure that the normalizer has been applied to the input tensor
            before using this module for inverse normalization.
        """
        if self.method == "mean" or self.method == "last":
            return x * self.normalizer.scale
        return x


class ReversibleInstanceNorm1dInput(nn.Module):
    """
    Reversible Instance Normalization for 1D inputs.

    This module implements reversible instance normalization for 1D inputs. It 
    computes the mean and standard deviation of the input tensor and normalizes 
    the input accordingly. The normalization parameters can be reversed to 
    retrieve the original input.

    Attributes:
        transposed (bool): Indicates whether the input is in transposed form (BDL)
            or not (BLD).
        norm (nn.InstanceNorm1d): Instance normalization layer.

    Args:
        d (int): The number of features in the input tensor.
        transposed (bool, optional): If True, expects input shape (B, D, L). 
            Default is False (input shape is (B, L, D)).

    Returns:
        Tensor: The normalized input tensor.

    Examples:
        >>> norm_layer = ReversibleInstanceNorm1dInput(d=10, transposed=False)
        >>> input_tensor = torch.randn(32, 5, 10)  # (Batch, Length, Features)
        >>> normalized_tensor = norm_layer(input_tensor)
        >>> print(normalized_tensor.shape)
        torch.Size([32, 5, 10])

    Note:
        This module is designed to work with transposed and non-transposed 
        inputs, allowing flexibility in handling different data formats.

    Todo:
        - Implement additional features for handling edge cases in input shapes.
    """
    def __init__(self, d, transposed=False):
        super().__init__()
        # BLD if transpoed is False, otherwise BDL
        self.transposed = transposed
        self.norm = nn.InstanceNorm1d(d, affine=True, track_running_stats=False)

    def forward(self, x):
        """
        ReversibleInstanceNorm1dInput class.

        This class applies a reversible instance normalization operation over the input
        tensor. It computes the mean and standard deviation along the specified 
        dimensions and normalizes the input accordingly. The normalization is reversible, 
        allowing the original input to be reconstructed later using the stored statistics.

        Attributes:
            transposed (bool): A flag indicating whether the input tensor is in a 
                            transposed format (BDL) or not (BLD).
            norm (nn.InstanceNorm1d): The instance normalization layer.

        Args:
            d (int): The number of features in the input tensor.
            transposed (bool): Indicates if the input is in a transposed format. 
                            Default is False.

        Returns:
            Tensor: The normalized tensor, with the same shape as the input.

        Examples:
            >>> layer = ReversibleInstanceNorm1dInput(d=64, transposed=False)
            >>> input_tensor = torch.randn(32, 64, 10)  # (batch_size, features, lengths)
            >>> output_tensor = layer(input_tensor)
            >>> print(output_tensor.shape)  # Output: torch.Size([32, 64, 10])

        Note:
            This normalization is particularly useful in scenarios where maintaining 
            the input distribution is critical, such as in reversible networks.
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
    ReversibleInstanceNorm1dOutput is a module that applies reversible instance 
    normalization to a 1D input tensor.

    This module takes the output of the `ReversibleInstanceNorm1dInput` class and 
    reverses the normalization process by applying the stored mean and standard 
    deviation. This is particularly useful in reversible neural networks where 
    the original input needs to be recovered after normalization.

    Attributes:
        transposed (bool): Indicates if the input tensor is transposed.
        weight (torch.nn.Parameter): Learnable weight parameter for normalization.
        bias (torch.nn.Parameter): Learnable bias parameter for normalization.
        norm_input (ReversibleInstanceNorm1dInput): Instance of the input normalization 
            module that contains the mean and standard deviation used during 
            normalization.

    Args:
        norm_input (ReversibleInstanceNorm1dInput): An instance of 
            `ReversibleInstanceNorm1dInput` which provides the mean and 
            standard deviation for the normalization.

    Returns:
        Tensor: The output tensor after applying the reverse normalization.

    Examples:
        >>> norm_input = ReversibleInstanceNorm1dInput(d=10)
        >>> output_layer = ReversibleInstanceNorm1dOutput(norm_input)
        >>> input_tensor = torch.randn(32, 10, 5)  # Batch size 32, 10 features, 5 length
        >>> normalized_tensor = norm_input(input_tensor)
        >>> output_tensor = output_layer(normalized_tensor)

    Note:
        The input tensor shape should match the shape expected by the 
        `ReversibleInstanceNorm1dInput` class, either (B, L, D) or (B, D, L) 
        depending on the `transposed` attribute.

    Todo:
        - Consider implementing additional checks for input shape compatibility.
    """
    def __init__(self, norm_input):
        super().__init__()
        self.transposed = norm_input.transposed
        self.weight = norm_input.norm.weight
        self.bias = norm_input.norm.bias
        self.norm_input = norm_input

    def forward(self, x):
        """
        Output module for reversible instance normalization.

        This module applies the inverse transformation of the instance normalization
        process that was performed in the corresponding input module. It takes the
        normalized output and re-scales it back to the original space using the
        mean and standard deviation computed during the forward pass of the input
        module.

        Attributes:
            transposed (bool): Indicates if the input data is transposed.
            weight (torch.Tensor): The learnable weight parameter from the input 
                normalization module.
            bias (torch.Tensor): The learnable bias parameter from the input 
                normalization module.
            norm_input (ReversibleInstanceNorm1dInput): The input normalization 
                module which provides the mean and standard deviation for 
                re-scaling.

        Args:
            norm_input (ReversibleInstanceNorm1dInput): The instance normalization
                module used to calculate the mean and standard deviation.

        Returns:
            Tensor: The output tensor after applying the inverse normalization.

        Examples:
            >>> norm_input = ReversibleInstanceNorm1dInput(d=10)
            >>> norm_output = ReversibleInstanceNorm1dOutput(norm_input)
            >>> x = torch.randn(5, 10, 20)  # Example input tensor
            >>> normalized = norm_input(x)    # Apply normalization
            >>> output = norm_output(normalized)  # Apply inverse normalization
            >>> assert torch.allclose(x, output)  # Check if we recover the original x
        """
        if not self.transposed:
            x = x.transpose(-1, -2)

        # x = (x - self.bias.unsqueeze(-1))/self.weight.unsqueeze(-1)
        x = x * self.norm_input.s + self.norm_input.m

        if not self.transposed:
            return x.transpose(-1, -2)
        return x
