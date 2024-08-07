# This code is derived from https://github.com/HazyResearch/state-spaces

"""Implementations of different types of residual functions."""

import torch
from torch import nn


class Residual(nn.Module):
    """
        Residual connection with constant affine weights.

    This class implements a residual connection that can simulate standard residual,
    no residual, and "constant gates" behaviors.

    Attributes:
        i_layer (int): The index of the current layer.
        d_input (int): The dimension of the input.
        d_model (int): The dimension of the model.
        alpha (float): The weight for the input in the residual connection.
        beta (float): The weight for the transformed input in the residual connection.

    Args:
        i_layer (int): The index of the current layer.
        d_input (int): The dimension of the input.
        d_model (int): The dimension of the model.
        alpha (float, optional): The weight for the input. Defaults to 1.0.
        beta (float, optional): The weight for the transformed input. Defaults to 1.0.

    Note:
        The input dimension (d_input) must be equal to the model dimension (d_model)
        unless alpha is set to 0.0.

    Examples:
        >>> residual = Residual(1, 64, 64)
        >>> x = torch.randn(10, 64)
        >>> y = torch.randn(10, 64)
        >>> output = residual(x, y, transposed=False)
        >>> output.shape
        torch.Size([10, 64])
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
        """
                Returns the output dimension of the residual connection.

        Returns:
            int: The dimension of the model (d_model), which is the output dimension
            of the residual connection.

        Note:
            This property always returns the d_model attribute, as the residual
            connection maintains the same dimension as the input model.

        Examples:
            >>> residual = Residual(1, 64, 64)
            >>> residual.d_output
            64
        """
        return self.d_model

    def forward(self, x, y, transposed):
        """
                Performs the forward pass of the residual connection.

        This method applies the residual connection by combining the input x
        and the transformed input y using the weights alpha and beta.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The transformed input tensor.
            transposed (bool): A flag indicating whether the input is transposed.
                This argument is not used in the current implementation but is
                included for consistency with other similar methods.

        Returns:
            torch.Tensor: The output of the residual connection.

        Note:
            The formula used is: output = alpha * x + beta * y
            If beta is 1.0, y is used as is. If alpha is 0, only y is returned.

        Examples:
            >>> residual = Residual(1, 64, 64, alpha=0.5, beta=1.5)
            >>> x = torch.randn(10, 64)
            >>> y = torch.randn(10, 64)
            >>> output = residual.forward(x, y, transposed=False)
            >>> output.shape
            torch.Size([10, 64])
        """
        y = self.beta * y if self.beta != 1.0 else y
        return self.alpha * x + y if self.alpha else y


class Affine(Residual):
    """
        Residual connection with learnable scalar multipliers on the main branch.

    This class extends the Residual class by adding learnable scalar multipliers
    to the main branch of the residual connection. It can be initialized with
    either a single scalar multiplier or one per dimension.

    Attributes:
        scalar (bool): If True, use a single scalar multiplier; otherwise, use one per dimension.
        gamma (float): Power factor for layer-dependent scaling initialization.
        affine (nn.Parameter): Learnable scalar multiplier(s) for the main branch.

    Args:
        *args: Variable length argument list passed to the parent Residual class.
        scalar (bool, optional): Whether to use a single scalar multiplier. Defaults to True.
        gamma (float, optional): Power factor for layer-dependent scaling. Defaults to 0.0.
        **kwargs: Arbitrary keyword arguments passed to the parent Residual class.

    Note:
        The affine parameter is initialized as:
        c * torch.ones(d), where
        c = beta * i_layer ** (-gamma)
        d = 1 if scalar is True, else d_input

    Examples:
        >>> affine_residual = Affine(1, 64, 64, scalar=False, gamma=0.1)
        >>> x = torch.randn(10, 64)
        >>> y = torch.randn(10, 64)
        >>> output = affine_residual(x, y, transposed=False)
        >>> output.shape
        torch.Size([10, 64])
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
                Performs the forward pass of the Affine residual connection.

        This method applies the residual connection by combining the input x
        and the transformed input y using the learnable affine parameter.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The transformed input tensor.
            transposed (bool): A flag indicating whether the input is transposed.
                If True, the affine parameter is unsqueezed along the last dimension.

        Returns:
            torch.Tensor: The output of the Affine residual connection.

        Note:
            The formula used is: output = alpha * x + c * y
            where c is the learnable affine parameter.

        Examples:
            >>> affine_residual = Affine(1, 64, 64, scalar=False, gamma=0.1)
            >>> x = torch.randn(10, 64)
            >>> y = torch.randn(10, 64)
            >>> output = affine_residual.forward(x, y, transposed=False)
            >>> output.shape
            torch.Size([10, 64])
        """
        c = self.affine
        if transposed:
            c = c.unsqueeze(-1)
        return self.alpha * x + c * y


class Feedforward(Residual):
    """
        A feedforward residual connection.

    This class implements a simple feedforward residual connection where the
    input x is ignored, and only the transformed input y is passed through.
    It's a special case of the Residual class with alpha set to 0.0 and beta set to 1.0.

    Args:
        *args: Variable length argument list passed to the parent Residual class.

    Note:
        This class effectively removes the residual connection, as it only
        passes through the transformed input y without adding the original input x.

    Examples:
        >>> feedforward = Feedforward(1, 64, 64)
        >>> x = torch.randn(10, 64)
        >>> y = torch.randn(10, 64)
        >>> output = feedforward(x, y, transposed=False)
        >>> output.shape
        torch.Size([10, 64])
        >>> torch.allclose(output, y)
        True
    """

    def __init__(self, *args):
        # print("Feedforward extra kwargs", kwargs)
        super().__init__(*args, alpha=0.0, beta=1.0)


class Highway(Residual):
    """
        Implements a Highway residual connection.

    This class extends the Residual class to create a Highway network connection,
    which allows the model to adaptively choose between passing information
    through a nonlinear transformation or passing it unchanged.

    Attributes:
        scaling_correction (float): A scaling factor to adjust the output.
        elemwise (bool): If True, uses element-wise multiplication for Wy;
                         otherwise, uses a linear transformation.
        Wx (nn.Linear): Linear transformation for the input x.
        Wy (nn.Parameter or nn.Linear): Transformation for the input y,
                                        either element-wise or linear.

    Args:
        *args: Variable length argument list passed to the parent Residual class.
        scaling_correction (bool, optional): If True, applies a scaling correction
                                             factor of 1.732. Defaults to False.
        elemwise (bool, optional): If True, uses element-wise multiplication
                                   for Wy. Defaults to False.

    Note:
        The Highway connection computes a gating mechanism to control
        information flow from the input and transformed input.

    Examples:
        >>> highway = Highway(1, 64, 64, scaling_correction=True, elemwise=False)
        >>> x = torch.randn(10, 64)
        >>> y = torch.randn(10, 64)
        >>> output = highway(x, y, transposed=False)
        >>> output.shape
        torch.Size([10, 64])
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
                Performs the forward pass of the Highway residual connection.

        This method implements the Highway network mechanism, which adaptively
        combines the original input and its transformation using a gating function.

        Args:
            x (torch.Tensor): The original input tensor.
            y (torch.Tensor): The transformed input tensor.
            transposed (bool): A flag indicating whether the input is transposed.
                This argument is not used in the current implementation but is
                included for consistency with other similar methods.

        Returns:
            torch.Tensor: The output of the Highway residual connection.

        Note:
            The Highway mechanism is computed as follows:
            1. Compute the gate: r = sigmoid(Wx(x) + Wy(y))
            2. Combine inputs: z = scaling_correction * (1 - r) * x + r * y

            If elemwise is True, Wy is applied as element-wise multiplication.
            Otherwise, it's applied as a linear transformation.

        Examples:
            >>> highway = Highway(1, 64, 64, scaling_correction=True, elemwise=False)
            >>> x = torch.randn(10, 64)
            >>> y = torch.randn(10, 64)
            >>> output = highway.forward(x, y, transposed=False)
            >>> output.shape
            torch.Size([10, 64])
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
        Implements a residual connection with depth-dependent decay.

    This class extends the Residual class to create a connection where the
    contribution of the residual path decays based on the depth of the layer.

    Attributes:
        power (float): The power factor used in the decay calculation.
        l2 (bool): If True, uses L2 normalization for alpha calculation;
                   otherwise, uses linear decay.

    Args:
        *args: Variable length argument list passed to the parent Residual class.
        power (float, optional): The power factor for decay calculation. Defaults to 0.5.
        l2 (bool, optional): Whether to use L2 normalization. Defaults to True.

    Note:
        The decay is calculated based on the layer index (i_layer) and the power factor.
        The contribution of the original input decreases as the network depth increases.

    Examples:
        >>> decay_residual = DecayResidual(1, 64, 64, power=0.5, l2=True)
        >>> x = torch.randn(10, 64)
        >>> y = torch.randn(10, 64)
        >>> output = decay_residual(x, y, transposed=False)
        >>> output.shape
        torch.Size([10, 64])
    """

    def __init__(self, *args, power=0.5, l2=True):
        # print("DecayResidual extra kwargs", kwargs)
        super().__init__(*args)
        self.power = power
        self.l2 = l2

    def forward(self, x, y, transposed):
        """
                Performs the forward pass of the DecayResidual connection.

        This method implements the depth-dependent decay mechanism for combining
        the original input and its transformation.

        Args:
            x (torch.Tensor): The original input tensor.
            y (torch.Tensor): The transformed input tensor.
            transposed (bool): A flag indicating whether the input is transposed.
                This argument is not used in the current implementation but is
                included for consistency with other similar methods.

        Returns:
            torch.Tensor: The output of the DecayResidual connection.

        Note:
            The decay mechanism is computed as follows:
            1. Calculate beta: beta = i_layer ** (-power)
            2. Calculate alpha:
               - If l2 is True: alpha = sqrt(1 - beta^2)
               - If l2 is False: alpha = 1 - beta
            3. Combine inputs: output = alpha * x + beta * y

            As the layer index increases, beta decreases, reducing the contribution
            of the transformed input y and increasing the contribution of the
            original input x.

        Examples:
            >>> decay_residual = DecayResidual(1, 64, 64, power=0.5, l2=True)
            >>> x = torch.randn(10, 64)
            >>> y = torch.randn(10, 64)
            >>> output = decay_residual.forward(x, y, transposed=False)
            >>> output.shape
            torch.Size([10, 64])
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
