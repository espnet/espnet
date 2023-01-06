# This code is derived from https://github.com/HazyResearch/state-spaces

""" Utility nn components,
in particular handling activations,initializations, and normalization layers
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract


def stochastic_depth(input: torch.tensor, p: float, mode: str, training: bool = True):
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first
                    one being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
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
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        # NOTE: need to upgrade to torchvision==0.11.0 to use StochasticDepth directly
        # from torchvision.ops import StochasticDepth
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "p=" + str(self.p)
        tmpstr += ", mode=" + str(self.mode)
        tmpstr += ")"
        return tmpstr


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
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
        """X: (batch, dim, lengths...)"""
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
    """Returns a linear nn.Module with control over axes order,
    initialization, and activation
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
    def forward(self, x):
        return F.relu(x) ** 2


class TransposedLinear(nn.Module):
    """Linear module on the second-to-last dimension
    Assumes shape (B, D, L), where L can be 1 or more axis
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
        num_axis = len(x.shape[2:])  # num_axis in L, for broadcasting bias
        y = contract("b u ..., v u -> b v ...", x, self.weight) + self.bias.view(
            -1, *[1] * num_axis
        )
        return y


class TransposedLN(nn.Module):
    """LayerNorm module over second dimension
    Assumes shape (B, D, L), where L can be 1 or more axis

    This is slow and a dedicated CUDA/Triton implementation
    shuld provide substantial end-to-end speedup
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
        assert self._name_ in ["layer", "none"]
        if self.transposed:
            x = x.unsqueeze(-1)
        x = self.forward(x)
        if self.transposed:
            x = x.squeeze(-1)
        return x


class TSNormalization(nn.Module):
    def __init__(self, method, horizon):
        super().__init__()

        self.method = method
        self.horizon = horizon

    def forward(self, x):
        # x must be BLD
        if self.method == "mean":
            self.scale = x.abs()[:, : -self.horizon].mean(dim=1)[:, None, :]
            return x / self.scale
        elif self.method == "last":
            self.scale = x.abs()[:, -self.horizon - 1][:, None, :]
            return x / self.scale
        return x


class TSInverseNormalization(nn.Module):
    def __init__(self, method, normalizer):
        super().__init__()

        self.method = method
        self.normalizer = normalizer

    def forward(self, x):
        if self.method == "mean" or self.method == "last":
            return x * self.normalizer.scale
        return x


class ReversibleInstanceNorm1dInput(nn.Module):
    def __init__(self, d, transposed=False):
        super().__init__()
        # BLD if transpoed is False, otherwise BDL
        self.transposed = transposed
        self.norm = nn.InstanceNorm1d(d, affine=True, track_running_stats=False)

    def forward(self, x):
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
    def __init__(self, norm_input):
        super().__init__()
        self.transposed = norm_input.transposed
        self.weight = norm_input.norm.weight
        self.bias = norm_input.norm.bias
        self.norm_input = norm_input

    def forward(self, x):
        if not self.transposed:
            x = x.transpose(-1, -2)

        # x = (x - self.bias.unsqueeze(-1))/self.weight.unsqueeze(-1)
        x = x * self.norm_input.s + self.norm_input.m

        if not self.transposed:
            return x.transpose(-1, -2)
        return x
