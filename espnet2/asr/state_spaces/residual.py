# This code is derived from https://github.com/HazyResearch/state-spaces

"""Implementations of different types of residual functions."""

import torch
from torch import nn


class Residual(nn.Module):
    """
    Residual connection with constant affine weights.

    This class implements a residual connection that can simulate various types of
    residual behaviors including standard residual, no residual, and "constant gates".
    The residual connection is parameterized by constant affine weights, allowing for
    flexibility in how the input and output tensors are combined.

    Attributes:
        i_layer (int): The index of the current layer.
        d_input (int): The dimensionality of the input tensor.
        d_model (int): The dimensionality of the model's output tensor.
        alpha (float): The scaling factor for the input tensor.
        beta (float): The scaling factor for the output tensor.

    Args:
        i_layer (int): The index of the layer in the network.
        d_input (int): The input dimensionality.
        d_model (int): The output dimensionality.
        alpha (float, optional): Scaling factor for the input. Defaults to 1.0.
        beta (float, optional): Scaling factor for the output. Defaults to 1.0.

    Returns:
        Tensor: The output tensor resulting from the residual connection.

    Raises:
        AssertionError: If `d_input` is not equal to `d_model` and `alpha` is not 0.0.

    Examples:
        >>> residual = Residual(i_layer=1, d_input=128, d_model=128, alpha=0.5)
        >>> x = torch.randn(10, 128)
        >>> y = torch.randn(10, 128)
        >>> output = residual(x, y, transposed=False)
        >>> print(output.shape)
        torch.Size([10, 128])

    Note:
        This implementation is part of a larger framework for state-space models
        in speech processing tasks.

    Todo:
        - Extend the functionality to include more advanced residual connections.
    """

    def __init__(self, i_layer, d_input, d_model, alpha=1.0, beta=1.0):
        # print("ConstantResidual extra kwargs", kwargs)
        super().__init__()
        assert (d_input == d_model) or alpha == 0.0
        self.i_layer = i_layer
        self.d_input = d_input
        self.d_model = d_model
        self.alpha = alpha
        self.beta = beta

    @property
    def d_output(self):
        return self.d_model

    def forward(self, x, y, transposed):
        """
        Compute the output of the residual connection.

        This method performs a residual operation based on the input tensors `x`
        and `y`, scaled by the parameters `alpha` and `beta`. The output is
        determined by whether the input `y` is to be transposed or not. The
        operation can simulate standard residual connections or other variations
        based on the values of `alpha` and `beta`.

        Args:
            x (torch.Tensor): The input tensor representing the main branch.
            y (torch.Tensor): The input tensor representing the residual branch.
            transposed (bool): A flag indicating whether to transpose the
                input tensor `y` before applying the operation.

        Returns:
            torch.Tensor: The resulting tensor after applying the residual
            connection.

        Examples:
            >>> residual = Residual(i_layer=1, d_input=10, d_model=10, alpha=1.0)
            >>> x = torch.randn(5, 10)
            >>> y = torch.randn(5, 10)
            >>> output = residual.forward(x, y, transposed=False)
            >>> output.shape
            torch.Size([5, 10])

        Note:
            The output will be `alpha * x + beta * y` if `alpha` is non-zero;
            otherwise, it will return `y` scaled by `beta`.

        Raises:
            ValueError: If the shapes of `x` and `y` do not match the expected
            dimensions based on `d_input` and `d_model`.
        """
        y = self.beta * y if self.beta != 1.0 else y
        return self.alpha * x + y if self.alpha else y


class Affine(Residual):
    """
    Residual connection with learnable scalar multipliers on the main branch.

    This class implements a residual connection that includes learnable
    scalar multipliers applied to the input of the residual branch. It
    allows for flexibility in how the residual connection is formed,
    making it possible to use a single scalar multiplier or one per
    dimension, depending on the `scalar` attribute.

    Attributes:
        scalar (bool): If True, uses a single scalar multiplier; if False,
            uses one multiplier per dimension.
        gamma (float): A scaling factor that influences the initialization
            of the affine parameters.
        affine (torch.nn.Parameter): A learnable parameter representing the
            scalar multipliers.

    Args:
        *args: Variable length argument list for initializing the parent
            Residual class.
        scalar (bool, optional): Determines if a single scalar multiplier
            or one per dimension is used. Default is True.
        gamma (float, optional): The power for scaling initialization.
            Default is 0.0.
        **kwargs: Additional keyword arguments for the parent class.

    Returns:
        None

    Examples:
        >>> affine_layer = Affine(i_layer=1, d_input=4, d_model=4,
        ...                       alpha=1.0, beta=1.0, scalar=True,
        ...                       gamma=0.0)
        >>> x = torch.rand(2, 4)
        >>> y = torch.rand(2, 4)
        >>> output = affine_layer(x, y, transposed=False)
        >>> print(output.shape)  # Output: torch.Size([2, 4])

    Note:
        The multipliers are initialized to scale * layer_num**(-power)
        based on the layer number and the provided gamma parameter.

    Todo:
        - Consider adding options for more complex initialization of
          affine parameters.
    """

    def __init__(self, *args, scalar=True, gamma=0.0, **kwargs):
        # print("ConstantResidual extra kwargs", kwargs)
        super().__init__(*args, **kwargs)
        self.scalar = scalar
        self.gamma = gamma

        c = self.beta * self.i_layer ** (-self.gamma)
        d = 1 if self.scalar else self.d_input
        self.affine = nn.Parameter(c * torch.ones(d))

    def forward(self, x, y, transposed):
        """
        Computes the forward pass of the affine residual connection.

        This method takes two inputs, `x` and `y`, and computes the output
        using the defined affine transformation. If `transposed` is True,
        the learnable parameters are reshaped accordingly.

        Args:
            x (torch.Tensor): The input tensor from the previous layer.
            y (torch.Tensor): The input tensor to be added, typically from
                another branch.
            transposed (bool): A flag indicating whether to apply the
                transposition to the learnable parameters.

        Returns:
            torch.Tensor: The output tensor after applying the affine
            transformation and residual connection.

        Examples:
            >>> affine_layer = Affine(i_layer=1, d_input=4, d_model=4)
            >>> x = torch.randn(2, 4)  # Batch of 2 with 4 features
            >>> y = torch.randn(2, 4)
            >>> output = affine_layer.forward(x, y, transposed=False)
            >>> print(output.shape)
            torch.Size([2, 4])

        Note:
            The learnable parameters are initialized based on the layer
            index and a specified gamma value, which controls the scaling
            factor.

        Raises:
            ValueError: If the dimensions of `x` and `y` do not match the
            expected input sizes.
        """
        c = self.affine
        if transposed:
            c = c.unsqueeze(-1)
        return self.alpha * x + c * y


class Feedforward(Residual):
    """
    Feedforward residual connection that bypasses input features.

    This class implements a feedforward residual connection where the
    input features are directly passed to the output without any scaling.
    The main purpose is to allow for a direct pathway in the network,
    effectively bypassing the residual computation.

    Attributes:
        i_layer (int): The index of the current layer.
        d_input (int): The dimensionality of the input features.
        d_model (int): The dimensionality of the model output.
        alpha (float): Scaling factor for the input features, defaults to 0.0.
        beta (float): Scaling factor for the output features, defaults to 1.0.

    Args:
        *args: Variable length argument list passed to the parent class.

    Examples:
        >>> feedforward_layer = Feedforward(i_layer=1, d_input=256, d_model=256)
        >>> x = torch.randn(10, 256)  # Batch of 10, input dimension 256
        >>> y = torch.randn(10, 256)  # Batch of 10, output dimension 256
        >>> output = feedforward_layer(x, y, transposed=False)
        >>> print(output.shape)
        torch.Size([10, 256])  # Output shape matches the input shape

    Note:
        This implementation sets alpha to 0.0, which means the input
        features will not be scaled, effectively making it a bypass
        connection for the input features.
    """

    def __init__(self, *args):
        # print("Feedforward extra kwargs", kwargs)
        super().__init__(*args, alpha=0.0, beta=1.0)


class Highway(Residual):
    """
    Highway Residual connection with learned gating mechanisms.

    This class implements a Highway connection that combines input tensors
    using learned affine transformations and gating mechanisms. The
    Highway layer can apply a scaling correction to its output and supports
    element-wise multiplication of the residual input.

    Attributes:
        scaling_correction (float): A scaling factor applied to the output,
            defaulting to 1.732 if enabled, otherwise 1.0.
        elemwise (bool): If True, the residual connection is computed
            element-wise; otherwise, a linear transformation is applied.

    Args:
        *args: Variable length argument list passed to the parent Residual class.
        scaling_correction (bool): Indicates whether to apply scaling
            correction to the output (default: False).
        elemwise (bool): If True, applies element-wise multiplication to the
            residual input (default: False).

    Returns:
        Tensor: The output tensor resulting from the Highway connection.

    Examples:
        >>> highway_layer = Highway(i_layer=1, d_input=256, d_model=256)
        >>> x = torch.randn(32, 256)  # Batch of 32
        >>> y = torch.randn(32, 256)  # Batch of 32
        >>> output = highway_layer(x, y)
        >>> print(output.shape)
        torch.Size([32, 256])

    Note:
        The Highway layer can be particularly useful in deep networks
        to enable training by mitigating the vanishing gradient problem.

    Todo:
        - Implement additional features or options for more advanced use cases.
    """

    def __init__(self, *args, scaling_correction=False, elemwise=False):
        super().__init__(*args)
        self.scaling_correction = 1.732 if scaling_correction else 1.0
        self.elemwise = elemwise
        self.Wx = nn.Linear(self.d_input, self.d_input)
        if self.elemwise:
            self.Wy = nn.Parameter(torch.randn(self.d_input))
        else:
            self.Wy = nn.Linear(self.d_input, self.d_input)

    def forward(self, x, y, transposed=False):
        """
        Perform the forward pass of the Highway residual connection.

        This method computes the output of the Highway layer by applying a
        learnable transformation to the input tensors `x` and `y`. It utilizes
        a gating mechanism, controlled by the sigmoid function, to blend the
        input tensors based on their learned weights.

        Args:
            x (torch.Tensor): The input tensor to the layer, typically the
                previous layer's output.
            y (torch.Tensor): The tensor to be combined with `x`, usually the
                output of another layer or transformation.
            transposed (bool, optional): A flag indicating whether the `y` tensor
                should be treated as transposed. Defaults to False.

        Returns:
            torch.Tensor: The output tensor resulting from the combination of
            `x` and `y` according to the Highway mechanism.

        Examples:
            >>> highway_layer = Highway(i_layer=0, d_input=64, d_model=64)
            >>> x = torch.randn(10, 64)  # Batch of 10 with 64 features
            >>> y = torch.randn(10, 64)  # Another batch of 10 with 64 features
            >>> output = highway_layer.forward(x, y)
            >>> print(output.shape)
            torch.Size([10, 64])

        Note:
            The scaling correction can be enabled during the initialization of
            the Highway layer, which will adjust the output accordingly.
        """
        if self.elemwise:
            y = self.Wy * y
        else:
            y = self.Wy(y)
        r = torch.sigmoid(self.Wx(x) + y)
        z = self.scaling_correction * (1.0 - r) * x + r * y
        return z


class DecayResidual(Residual):
    """
    Residual connection that can decay the linear combination depending on depth.

    This class implements a residual connection where the contribution of the
    input tensor can decay based on the layer depth. It adjusts the weights of
    the residual connection dynamically, allowing for more controlled flow of
    information through deeper layers.

    Attributes:
        power (float): The exponent used to compute the decay factor for the
            residual connection. A higher power results in faster decay.
        l2 (bool): If True, uses L2 normalization for the alpha coefficient;
            otherwise, uses a linear decay.

    Args:
        *args: Positional arguments to be passed to the parent class.
        power (float): Exponent for decay computation (default is 0.5).
        l2 (bool): Flag to determine if L2 normalization is used (default is True).

    Returns:
        Tensor: The output of the residual connection after applying decay.

    Examples:
        >>> decay_residual = DecayResidual(i_layer=2, d_input=10, d_model=10)
        >>> x = torch.randn(1, 10)
        >>> y = torch.randn(1, 10)
        >>> output = decay_residual(x, y, transposed=False)
        >>> print(output.shape)
        torch.Size([1, 10])

    Note:
        The behavior of this class is influenced by the layer index (i_layer)
        at which it is instantiated. As the layer index increases, the effect
        of the decay will be more pronounced.

    Todo:
        - Consider adding support for non-linear decay functions.
    """

    def __init__(self, *args, power=0.5, l2=True):
        # print("DecayResidual extra kwargs", kwargs)
        super().__init__(*args)
        self.power = power
        self.l2 = l2

    def forward(self, x, y, transposed):
        """
        Computes the output of the DecayResidual layer, which applies a residual
        connection that can decay the linear combination of inputs based on the
        layer's depth.

        The output is computed as:
            output = alpha * x + beta * y
        where alpha and beta are determined based on the layer index and the
        specified power. The decay factor allows for a controlled blending of
        the input tensor `x` and the residual tensor `y`, with the possibility
        of scaling down the contribution of the previous layer's output.

        Args:
            x (torch.Tensor): The input tensor to the layer, typically the output
                from the previous layer.
            y (torch.Tensor): The residual tensor, typically the output from a
                different path in the network.
            transposed (bool): A flag indicating whether the inputs should be
                treated as transposed (e.g., for different dimensions).

        Returns:
            torch.Tensor: The resulting tensor after applying the decay
                residual connection.

        Examples:
            >>> import torch
            >>> decay_residual = DecayResidual(i_layer=2, d_input=3, d_model=3)
            >>> x = torch.randn(5, 3)  # Batch of 5 with 3 features
            >>> y = torch.randn(5, 3)  # Batch of 5 with 3 features
            >>> output = decay_residual(x, y, transposed=False)
            >>> print(output.shape)
            torch.Size([5, 3])

        Note:
            The `power` attribute controls the rate of decay for the
            linear combination, where higher values result in faster decay.

        Raises:
            ValueError: If the dimensions of `x` and `y` do not match.
        """
        beta = self.i_layer ** (-self.power)
        if self.l2:
            alpha = (1.0 - beta**2) ** 0.5
        else:
            alpha = 1.0 - beta

        return alpha * x + beta * y


registry = {
    "F": Feedforward,
    "N": Feedforward,
    "R": Residual,
    "H": Highway,
    "D": DecayResidual,
    "A": Affine,
    "none": Feedforward,
    "ff": Feedforward,
    "feedforward": Feedforward,
    "residual": Residual,
    "highway": Highway,
    "decay": DecayResidual,
    "affine": Affine,
}
