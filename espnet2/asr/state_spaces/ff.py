# This code is derived from https://github.com/HazyResearch/state-spaces

""" Implementation of FFN block in the style of Transformers """

from functools import partial

from torch import nn

from espnet2.asr.state_spaces.base import SequenceModule
from espnet2.asr.state_spaces.components import (
    DropoutNd,
    LinearActivation,
)


class FF(SequenceModule):
    def __init__(
        self,
        d_input,
        expand=2,
        d_output=None,
        transposed=False,
        activation="gelu",
        initializer=None,
        dropout=0.0,
        tie_dropout=False,
    ):
        super().__init__()
        self.d_output = d_input if d_output is None else d_output
        self.transposed = transposed
        d_inner = expand * d_input

        linear1 = LinearActivation(
            d_input,
            d_inner,
            transposed=transposed,
            activation=activation,
            initializer=initializer,
            activate=True,
        )
        dropout_cls = (
            partial(DropoutNd, transposed=self.transposed)
            if tie_dropout
            else nn.Dropout
        )
        # dropout_cls = nn.Dropout2d if self.transposed else nn.Dropout
        drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        linear2 = LinearActivation(
            d_inner,
            self.d_output,
            transposed=transposed,
            activation=None,
            initializer=initializer,
            activate=False,
        )

        self.ff = nn.Sequential(
            linear1,
            drop,
            linear2,
        )

    def forward(self, x, *args, **kwargs):
        return self.ff(x), None

    def step(self, x, state, **kwargs):
        # x: [batch, d_input]
        if self.transposed:
            # expects: [batch, d_input, seq_len]
            return self.ff(x.unsqueeze(-1)).squeeze(-1), state
        else:
            return self.ff(x), state
