"""Activation functions for Transducer models."""

import torch
from packaging.version import parse as V


def get_activation(
    activation_type: str,
    ftswish_threshold: float = -0.2,
    ftswish_mean_shift: float = 0.0,
    hardtanh_min_val: int = -1.0,
    hardtanh_max_val: int = 1.0,
    leakyrelu_neg_slope: float = 0.01,
    smish_alpha: float = 1.0,
    smish_beta: float = 1.0,
    softplus_beta: float = 1.0,
    softplus_threshold: int = 20,
    swish_beta: float = 1.0,
) -> torch.nn.Module:
    """
    Return the specified activation function as a PyTorch module.

    This function provides a way to obtain various activation functions 
    based on the specified type. It supports standard activations such 
    as ReLU, Tanh, and others, as well as custom formulations like 
    FTSwish, Mish, Smish, and Swish. The parameters can be adjusted 
    to customize the behavior of certain activation functions.

    Args:
        activation_type (str): The type of activation function to return. 
            Options include: 'ftswish', 'hardtanh', 'leaky_relu', 
            'mish', 'relu', 'selu', 'smish', 'swish', 'tanh', 
            'identity'.
        ftswish_threshold (float): Threshold value for FTSwish activation formulation.
        ftswish_mean_shift (float): Mean shifting value for FTSwish activation formulation.
        hardtanh_min_val (int): Minimum value of the linear region range for HardTanh.
        hardtanh_max_val (int): Maximum value of the linear region range for HardTanh.
        leakyrelu_neg_slope (float): Negative slope value for LeakyReLU activation.
        smish_alpha (float): Alpha value for Smish activation formulation.
        smish_beta (float): Beta value for Smish activation formulation.
        softplus_beta (float): Beta value for softplus activation formulation in Mish.
        softplus_threshold (int): Values above this revert to a linear function in Mish.
        swish_beta (float): Beta value for Swish variant formulation.

    Returns:
        torch.nn.Module: A PyTorch activation function module corresponding to the 
        specified activation_type.

    Raises:
        KeyError: If the specified activation_type is not recognized.

    Examples:
        >>> activation = get_activation('relu')
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> activation(x)
        tensor([0., 0., 1.])

        >>> activation = get_activation('ftswish', ftswish_threshold=-0.1)
        >>> activation(x)
        tensor([-0.0000, 0.0000, 1.0000])

    Note:
        Ensure that the chosen activation function is compatible with 
        your model architecture. For custom functions like FTSwish, 
        additional parameters may be required for proper behavior.
    """
    torch_version = V(torch.__version__)

    activations = {
        "ftswish": (
            FTSwish,
            {"threshold": ftswish_threshold, "mean_shift": ftswish_mean_shift},
        ),
        "hardtanh": (
            torch.nn.Hardtanh,
            {"min_val": hardtanh_min_val, "max_val": hardtanh_max_val},
        ),
        "leaky_relu": (torch.nn.LeakyReLU, {"negative_slope": leakyrelu_neg_slope}),
        "mish": (
            Mish,
            {
                "softplus_beta": softplus_beta,
                "softplus_threshold": softplus_threshold,
                "use_builtin": torch_version >= V("1.9"),
            },
        ),
        "relu": (torch.nn.ReLU, {}),
        "selu": (torch.nn.SELU, {}),
        "smish": (Smish, {"alpha": smish_alpha, "beta": smish_beta}),
        "swish": (
            Swish,
            {"beta": swish_beta, "use_builtin": torch_version >= V("1.8")},
        ),
        "tanh": (torch.nn.Tanh, {}),
        "identity": (torch.nn.Identity, {}),
    }

    act_func, act_args = activations[activation_type]

    return act_func(**act_args)


class FTSwish(torch.nn.Module):
    """
    Flatten-T Swish activation definition.

    FTSwish(x) = x * sigmoid(x) + threshold
                  where FTSwish(x) < 0 = threshold.

    This activation function is designed to provide a smooth transition 
    for values below a specified threshold, allowing for improved 
    gradient flow during training. It can be particularly useful in 
    neural networks for tasks such as speech recognition.

    Reference: https://arxiv.org/abs/1812.06247

    Args:
        threshold (float): Threshold value for FTSwish activation 
            formulation. Must be less than 0.
        mean_shift (float): Mean shifting value for FTSwish activation 
            formulation. Applied only if not equal to 0 (disabled by 
            default).

    Examples:
        >>> ftswish = FTSwish(threshold=-0.5, mean_shift=0.1)
        >>> input_tensor = torch.tensor([-1.0, 0.0, 1.0])
        >>> output_tensor = ftswish(input_tensor)
        >>> print(output_tensor)
        tensor([-0.5000,  0.5000,  1.0000])

    Raises:
        AssertionError: If the threshold is not less than 0.
    """

    def __init__(self, threshold: float = -0.2, mean_shift: float = 0) -> None:
        super().__init__()

        assert threshold < 0, "FTSwish threshold parameter should be < 0."

        self.threshold = threshold
        self.mean_shift = mean_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Activation functions for Transducer models.

        This module provides various activation functions, including FTSwish, Mish,
        Smish, and Swish, suitable for use in transducer models. Each activation
        function is defined as a class that inherits from torch.nn.Module, allowing
        for seamless integration into PyTorch models.

        The main function, `get_activation`, returns the desired activation function
        based on the provided type and parameters.

        Attributes:
            - FTSwish: Flatten-T Swish activation function.
            - Mish: Mish activation function.
            - Smish: Smish activation function.
            - Swish: Swish activation function.

        Args:
            activation_type (str): The type of activation function to retrieve.
            ftswish_threshold (float): Threshold value for FTSwish activation 
                formulation (default: -0.2).
            ftswish_mean_shift (float): Mean shifting value for FTSwish 
                activation (default: 0.0).
            hardtanh_min_val (int): Minimum value for HardTanh (default: -1.0).
            hardtanh_max_val (int): Maximum value for HardTanh (default: 1.0).
            leakyrelu_neg_slope (float): Negative slope for LeakyReLU 
                (default: 0.01).
            smish_alpha (float): Alpha value for Smish activation (default: 1.0).
            smish_beta (float): Beta value for Smish activation (default: 1.0).
            softplus_beta (float): Beta value for softplus in Mish (default: 1.0).
            softplus_threshold (int): Threshold for softplus in Mish 
                (default: 20).
            swish_beta (float): Beta value for Swish (default: 1.0).

        Returns:
            torch.nn.Module: The specified activation function as a 
            PyTorch module.

        Examples:
            >>> activation = get_activation('smish', smish_alpha=1.0, smish_beta=1.0)
            >>> output = activation(torch.tensor([-1.0, 0.0, 1.0]))

        Raises:
            ValueError: If the specified activation type is not supported.

        Note:
            Ensure that the input tensor is of type torch.Tensor when 
            using the activation functions.
        """
        x = (x * torch.sigmoid(x)) + self.threshold
        x = torch.where(x >= 0, x, torch.tensor([self.threshold], device=x.device))

        if self.mean_shift != 0:
            x.sub_(self.mean_shift)

        return x


class Mish(torch.nn.Module):
    """
    Mish activation definition.

    The Mish activation function is defined as:
    
        Mish(x) = x * tanh(softplus(x))
    
    Where softplus is defined as:
    
        softplus(x) = log(1 + exp(x))

    This activation function has been shown to improve the performance of deep 
    neural networks in various tasks.

    Reference: https://arxiv.org/abs/1908.08681.

    Args:
        softplus_beta (float): Beta value for the softplus activation formulation.
            Typically, this should satisfy the condition 0 < softplus_beta < 2.
        softplus_threshold (float): Values above this threshold revert to a linear 
            function. Typically, it should satisfy the condition 10 < 
            softplus_threshold < 20.
        use_builtin (bool): Flag to indicate whether to use the built-in PyTorch 
            Mish activation function if available (introduced in PyTorch 1.9).

    Examples:
        >>> mish = Mish(softplus_beta=1.0, softplus_threshold=20)
        >>> input_tensor = torch.tensor([-1.0, 0.0, 1.0])
        >>> output_tensor = mish(input_tensor)
        >>> print(output_tensor)
        tensor([-0.3133, 0.0000, 0.7311])

    Note:
        The Mish activation is continuously differentiable and non-monotonic, 
        which can lead to better training dynamics.

    Raises:
        AssertionError: If `softplus_beta` is not in the valid range or if 
        `softplus_threshold` is not in the valid range.
    """

    def __init__(
        self,
        softplus_beta: float = 1.0,
        softplus_threshold: int = 20,
        use_builtin: bool = False,
    ) -> None:
        super().__init__()

        if use_builtin:
            self.mish = torch.nn.Mish()
        else:
            self.tanh = torch.nn.Tanh()
            self.softplus = torch.nn.Softplus(
                beta=softplus_beta, threshold=softplus_threshold
            )

            self.mish = lambda x: x * self.tanh(self.softplus(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Activation functions for Transducer models.

        This module provides various activation functions, including FTSwish, 
        Mish, Smish, and Swish, which can be utilized in neural network 
        architectures. The activation functions are implemented as subclasses 
        of `torch.nn.Module` and can be instantiated with configurable 
        parameters.

        Classes:
            - FTSwish: Implements the Flatten-T Swish activation function.
            - Mish: Implements the Mish activation function.
            - Smish: Implements the Smish activation function.
            - Swish: Implements the Swish activation function.

        Example usage:
            # Create an instance of FTSwish
            ftswish = FTSwish(threshold=-0.1, mean_shift=0.5)

            # Apply the activation function to a tensor
            input_tensor = torch.tensor([-1.0, 0.0, 1.0])
            output_tensor = ftswish(input_tensor)
            print(output_tensor)

            # Get a specific activation function using the helper function
            activation_function = get_activation("mish", softplus_beta=1.0)
            output_tensor = activation_function(input_tensor)
            print(output_tensor)
                """
        return self.mish(x)


class Smish(torch.nn.Module):
    """
    Activation functions for Transducer models.

    This module provides various activation functions, including FTSwish, 
    Mish, Smish, and Swish, suitable for use in transducer models. Each 
    activation function is defined as a class that inherits from 
    torch.nn.Module, allowing for seamless integration into PyTorch models.

    The main function, `get_activation`, returns the desired activation 
    function based on the provided type and parameters.

    Attributes:
        - FTSwish: Flatten-T Swish activation function.
        - Mish: Mish activation function.
        - Smish: Smish activation function.
        - Swish: Swish activation function.

    Args:
        activation_type (str): The type of activation function to retrieve.
        ftswish_threshold (float): Threshold value for FTSwish activation 
            formulation (default: -0.2).
        ftswish_mean_shift (float): Mean shifting value for FTSwish 
            activation (default: 0.0).
        hardtanh_min_val (float): Minimum value for HardTanh (default: -1.0).
        hardtanh_max_val (float): Maximum value for HardTanh (default: 1.0).
        leakyrelu_neg_slope (float): Negative slope for LeakyReLU 
            (default: 0.01).
        smish_alpha (float): Alpha value for Smish activation (default: 1.0).
        smish_beta (float): Beta value for Smish activation (default: 1.0).
        softplus_beta (float): Beta value for softplus in Mish (default: 1.0).
        softplus_threshold (int): Threshold for softplus in Mish 
            (default: 20).
        swish_beta (float): Beta value for Swish (default: 1.0).

    Returns:
        torch.nn.Module: The specified activation function as a 
        PyTorch module.

    Examples:
        >>> activation = get_activation('smish', smish_alpha=1.0, smish_beta=1.0)
        >>> output = activation(torch.tensor([-1.0, 0.0, 1.0]))

    Raises:
        ValueError: If the specified activation type is not supported.

    Note:
        Ensure that the input tensor is of type torch.Tensor when 
        using the activation functions.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()

        self.tanh = torch.nn.Tanh()

        self.alpha = alpha if alpha > 0 else 1
        self.beta = beta if beta > 0 else 1

        self.smish = lambda x: (self.alpha * x) * self.tanh(
            torch.log(1 + torch.sigmoid((self.beta * x)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Activation functions for Transducer models.

        This module provides various activation functions, including FTSwish, 
        Mish, Smish, and Swish, suitable for use in transducer models. Each 
        activation function is defined as a class that inherits from 
        torch.nn.Module, allowing for seamless integration into PyTorch models.

        The main function, `get_activation`, returns the desired activation 
        function based on the provided type and parameters.

        Attributes:
            - FTSwish: Flatten-T Swish activation function.
            - Mish: Mish activation function.
            - Smish: Smish activation function.
            - Swish: Swish activation function.

        Args:
            activation_type (str): The type of activation function to retrieve.
            ftswish_threshold (float): Threshold value for FTSwish activation 
                formulation (default: -0.2).
            ftswish_mean_shift (float): Mean shifting value for FTSwish 
                activation (default: 0.0).
            hardtanh_min_val (int): Minimum value for HardTanh (default: -1.0).
            hardtanh_max_val (int): Maximum value for HardTanh (default: 1.0).
            leakyrelu_neg_slope (float): Negative slope for LeakyReLU 
                (default: 0.01).
            smish_alpha (float): Alpha value for Smish activation (default: 1.0).
            smish_beta (float): Beta value for Smish activation (default: 1.0).
            softplus_beta (float): Beta value for softplus in Mish (default: 1.0).
            softplus_threshold (int): Threshold for softplus in Mish 
                (default: 20).
            swish_beta (float): Beta value for Swish (default: 1.0).

        Returns:
            torch.nn.Module: The specified activation function as a 
            PyTorch module.

        Examples:
            >>> activation = get_activation('smish', smish_alpha=1.0, smish_beta=1.0)
            >>> output = activation(torch.tensor([-1.0, 0.0, 1.0]))

        Raises:
            ValueError: If the specified activation type is not supported.

        Note:
            Ensure that the input tensor is of type torch.Tensor when 
            using the activation functions.
        """
        return self.smish(x)


class Swish(torch.nn.Module):
    """
    Swish activation definition.

    Swish(x) = (beta * x) * sigmoid(x), where beta = 1 defines standard Swish
    activation.

    References:
        - https://arxiv.org/abs/2108.12943
        - https://arxiv.org/abs/1710.05941
        - E-swish variant: https://arxiv.org/abs/1801.07145

    Attributes:
        beta (float): Beta parameter for E-Swish. Should be greater than or equal to
        1. If beta < 1, standard Swish is used.
        use_builtin (bool): Whether to use the built-in PyTorch function if available.

    Args:
        beta: Beta parameter for E-Swish activation. (beta >= 1)
        use_builtin: If True, utilize the built-in PyTorch SiLU function if available.

    Examples:
        >>> swish = Swish(beta=1.0)
        >>> input_tensor = torch.tensor([0.0, 1.0, 2.0])
        >>> output_tensor = swish(input_tensor)
        >>> print(output_tensor)
        tensor([0.0000, 0.7311, 1.7616])

        >>> swish_e = Swish(beta=0.5, use_builtin=True)
        >>> output_tensor_e = swish_e(input_tensor)
        >>> print(output_tensor_e)
        tensor([0.0000, 0.5000, 1.0000])

    Note:
        The Swish activation function has been shown to improve performance
        in certain deep learning tasks compared to traditional activation
        functions like ReLU.

    Raises:
        ValueError: If beta is less than 1.
    """

    def __init__(self, beta: float = 1.0, use_builtin: bool = False) -> None:
        super().__init__()

        self.beta = beta

        if beta > 1:
            self.swish = lambda x: (self.beta * x) * torch.sigmoid(x)
        else:
            if use_builtin:
                self.swish = torch.nn.SiLU()
            else:
                self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Activation functions for Transducer models.

        This module provides various activation functions, including FTSwish, Mish,
        Smish, and Swish, suitable for use in transducer models. Each activation
        function is defined as a class that inherits from torch.nn.Module, allowing
        for seamless integration into PyTorch models.

        The main function, `get_activation`, returns the desired activation function
        based on the provided type and parameters.

        Attributes:
            - FTSwish: Flatten-T Swish activation function.
            - Mish: Mish activation function.
            - Smish: Smish activation function.
            - Swish: Swish activation function.

        Args:
            activation_type (str): The type of activation function to retrieve.
            ftswish_threshold (float): Threshold value for FTSwish activation 
                formulation (default: -0.2).
            ftswish_mean_shift (float): Mean shifting value for FTSwish 
                activation (default: 0.0).
            hardtanh_min_val (int): Minimum value for HardTanh (default: -1.0).
            hardtanh_max_val (int): Maximum value for HardTanh (default: 1.0).
            leakyrelu_neg_slope (float): Negative slope for LeakyReLU 
                (default: 0.01).
            smish_alpha (float): Alpha value for Smish activation (default: 1.0).
            smish_beta (float): Beta value for Smish activation (default: 1.0).
            softplus_beta (float): Beta value for softplus in Mish (default: 1.0).
            softplus_threshold (int): Threshold for softplus in Mish 
                (default: 20).
            swish_beta (float): Beta value for Swish (default: 1.0).

        Returns:
            torch.nn.Module: The specified activation function as a 
            PyTorch module.

        Examples:
            >>> activation = get_activation('smish', smish_alpha=1.0, smish_beta=1.0)
            >>> output = activation(torch.tensor([-1.0, 0.0, 1.0]))

        Raises:
            ValueError: If the specified activation type is not supported.

        Note:
            Ensure that the input tensor is of type torch.Tensor when 
            using the activation functions.
        """
        return self.swish(x)
