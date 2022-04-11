"""Activation functions for Transducer."""

from distutils.version import LooseVersion

import torch


def get_activation(
    activation_type: str,
    hardtanh_min_val: int = -1.0,
    hardtanh_max_val: int = 1.0,
    smish_alpha: float = 1.0,
    smish_beta: float = 1.0,
    softplus_beta: float = 1.0,
    softplus_threshold: int = 20,
    swish_beta: float = 1.0,
) -> torch.nn.Module:
    """Return activation function.

    Args:
        activation_type: Activation type.
        hardtanh_min_val: Minimum value of the linear region range for HardTanh.
        hardtanh_max_val: Maximum value of the linear region range for HardTanh.
        smish_alpha: Alpha value for Smish variant fomulation.
        smish_beta: Beta value for Smish variant formulation.
        softplus_beta: Beta value for softplus formulation in Mish.
        softplus_threshold: Values above this revert to a linear function in Mish.
        swish_beta: Beta value for E-Swish variant formulation.

    Returns:
        : Activation function.

    """

    torch_version = LooseVersion(torch.__version__)

    activation_funcs = {
        "gcu": GCU(),
        "hardtanh": torch.nn.Hardtanh(hardtanh_min_val, hardtanh_max_val),
        "listh": LiSTH(),
        "mish": Mish(
            softplus_beta, softplus_threshold, torch_version >= LooseVersion("1.9")
        ),
        "relu": torch.nn.ReLU(),
        "selu": torch.nn.SELU(),
        "smish": Smish(smish_alpha, smish_beta),
        "swish": Swish(swish_beta, torch_version >= LooseVersion("1.8")),
        "tanh": torch.nn.Tanh(),
    }

    return activation_funcs[activation_type]


class GCU(torch.nn.Module):
    """Growing Cosine Unit definition.

    Reference: https://arxiv.org/abs/2108.12943.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        return x * torch.cos(x)


class LiSTH(torch.nn.Module):
    """LiSTH activation definition.

    Reference: https://arxiv.org/abs/1901.05894.

    """

    def __init__(self):
        super().__init__()

        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        return x * self.tanh(x)


class Mish(torch.nn.Module):
    """Mish activation definition.

    Reference: https://arxiv.org/abs/1908.08681.

    Args:
        beta: Beta value for softplus formulation.
        threshold: Values above this revert to a linear function.

    """

    def __init__(self, beta=1, threshold=20, use_builtin=False):
        super().__init__()

        if use_builtin:
            self.mish = torch.nn.Mish()
        else:
            self.tanh = torch.nn.Tanh()
            self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)

            self.mish = lambda x: x * self.tanh(self.softplus(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        return self.mish(x)


class Smish(torch.nn.Module):
    """Smish activation definition.

    Reference: https://www.mdpi.com/2079-9292/11/4/540/htm.

    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()

        self.tanh = torch.nn.Tanh()

        self.alpha = alpha
        self.beta = beta

        self.smish = lambda x: (self.alpha * x) * self.tanh(
            torch.log(1 + torch.sigmoid((self.beta * x)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        return self.smish(x)


class Swish(torch.nn.Module):
    """Swish activation definition.

    References:
        https://arxiv.org/abs/2108.12943 / https://arxiv.org/abs/1710.05941v1.
        E-variant: https://arxiv.org/abs/1801.07145.

    Args:
        beta: Beta parameter for E-Swish variant.

    """

    def __init__(self, beta=1, use_builtin=False):
        super().__init__()

        if use_builtin:
            self.swish = torch.nn.SiLU()
        else:
            self.beta = beta

            self.swish = lambda x: (self.beta * x) * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        return self.swish(x)
