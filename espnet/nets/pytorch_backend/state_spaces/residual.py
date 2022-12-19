# This code is derived from https://github.com/HazyResearch/state-spaces

""" Implementations of different types of residual functions. """

import torch
from torch import nn


class Residual(nn.Module):
    """Residual connection with constant affine weights. Can simulate standard residual, no residual, and "constant gates"."""

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

    def forward(self, x, y, transposed):  # TODO documentation of transposed
        y = self.beta * y if self.beta != 1.0 else y
        return self.alpha * x + y if self.alpha else y


class Affine(Residual):
    """Residual connection with learnable scalar multipliers on the main branch
    scalar: Single scalar multiplier, or one per dimension
    scale, power: Initialize to scale * layer_num**(-power)
    """

    def __init__(self, *args, scalar=True, gamma=0.0, **kwargs):
        # print("ConstantResidual extra kwargs", kwargs)
        super().__init__(*args, **kwargs)
        self.scalar = scalar
        self.gamma = gamma

        c = self.beta * self.i_layer ** (-self.gamma)
        d = 1 if self.scalar else self.d_input
        self.affine = nn.Parameter(c * torch.ones(d))

    def forward(self, x, y, transposed):  # TODO documentation of transposed
        c = self.affine
        if transposed:
            c = c.unsqueeze(-1)
        return self.alpha * x + c * y


class Feedforward(Residual):
    def __init__(self, *args):
        # print("Feedforward extra kwargs", kwargs)
        super().__init__(*args, alpha=0.0, beta=1.0)


class Highway(Residual):
    def __init__(self, *args, scaling_correction=False, elemwise=False):
        super().__init__(*args)
        self.scaling_correction = 1.732 if scaling_correction else 1.0  # TODO
        self.elemwise = elemwise
        self.Wx = nn.Linear(self.d_input, self.d_input)
        if self.elemwise:
            self.Wy = nn.Parameter(torch.randn(self.d_input))
        else:
            self.Wy = nn.Linear(self.d_input, self.d_input)

    def forward(self, x, y, transposed=False):  # TODO handle this case
        if self.elemwise:
            y = self.Wy * y
        else:
            y = self.Wy(y)
        r = torch.sigmoid(self.Wx(x) + y)
        z = self.scaling_correction * (1.0 - r) * x + r * y
        return z


class DecayResidual(Residual):
    """Residual connection that can decay the linear combination depending on depth."""

    def __init__(self, *args, power=0.5, l2=True):
        # print("DecayResidual extra kwargs", kwargs)
        super().__init__(*args)
        self.power = power
        self.l2 = l2

    def forward(self, x, y, transposed):
        beta = self.i_layer ** (-self.power)
        if self.l2:
            alpha = (1.0 - beta ** 2) ** 0.5
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
