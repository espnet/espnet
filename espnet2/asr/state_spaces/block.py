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
    """Residual block wrapper for black box layer.

    The SequenceResidualBlock class implements a generic
    (batch, length, d_input) -> (batch, length, d_input) transformation

    Args:
        d_input: Input feature dimension
        i_layer: Layer index, only needs to be passed into certain residuals like Decay
        dropout: Dropout for black box module
        tie_dropout: Tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d
        transposed: Transpose inputs so each layer receives (batch, dim, length)
        layer: Config for black box module
        residual: Config for residual function
        norm: Config for normalization layer
        pool: Config for pooling layer per stage
        drop_path: Drop ratio for stochastic depth
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
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, state=None, **kwargs):
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
