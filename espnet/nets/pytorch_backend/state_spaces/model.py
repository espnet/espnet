# This code is derived from https://github.com/HazyResearch/state-spaces

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from espnet.nets.pytorch_backend.state_spaces.base import SequenceModule
from espnet.nets.pytorch_backend.state_spaces.block import SequenceResidualBlock
from espnet.nets.pytorch_backend.state_spaces.components import DropoutNd, Normalization
from espnet.nets.pytorch_backend.state_spaces.utils import to_dict, to_list


class SequenceModel(SequenceModule):
    """Isotropic deep sequence model backbone, in the style of ResNets / Transformers.

    The SequenceModel class implements a generic
    (batch, length, d_input) -> (batch, length, d_output) transformation

    Args:
        d_model: Resize input (useful for deep models with residuals)
        n_layers: Number of layers
        transposed: Transpose inputs so each layer receives (batch, dim, length)
        dropout: Dropout parameter applied on every residual and every layer
        tie_dropout: Tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d
        prenorm: Pre-norm vs. post-norm
        n_repeat: Each layer is repeated n times per stage before applying pooling
        layer: Layer config, must be specified
        residual: Residual config
        norm: Normalization config (e.g. layer vs batch)
        pool: Config for pooling layer per stage
        track_norms: Log norms of each layer output
        dropinp: Input dropout
        drop_path: Stochastic depth for each residual path
    """

    def __init__(
        self,
        d_model,
        n_layers=1,
        transposed=False,
        dropout=0.0,
        tie_dropout=False,
        prenorm=True,
        n_repeat=1,
        layer=None,
        residual=None,
        norm=None,
        pool=None,
        track_norms=True,
        dropinp=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        # Save arguments needed for forward pass
        self.d_model = d_model
        self.transposed = transposed
        self.track_norms = track_norms

        # Input dropout (not really used)
        dropout_fn = (
            partial(DropoutNd, transposed=self.transposed)
            if tie_dropout
            else nn.Dropout
        )
        self.drop = dropout_fn(dropinp) if dropinp > 0.0 else nn.Identity()
        layer = to_list(layer, recursive=False)

        # Some special arguments are passed into each layer
        for _layer in layer:
            # If layers don't specify dropout, add it
            if _layer.get("dropout", None) is None:
                _layer["dropout"] = dropout
            # Ensure all layers are shaped the same way
            _layer["transposed"] = transposed

        # Duplicate layers
        layers = layer * n_layers * n_repeat

        # Instantiate layers
        _layers = []
        d = d_model
        for i, layer in enumerate(layers):
            # Pool at the end of every n_repeat blocks
            pool_cfg = pool if (i + 1) % n_repeat == 0 else None
            block = SequenceResidualBlock(
                d,
                i + 1,
                prenorm=prenorm,
                dropout=dropout,
                tie_dropout=tie_dropout,
                transposed=transposed,
                layer=layer,
                residual=residual,
                norm=norm,
                pool=pool_cfg,
                drop_path=drop_path,
            )
            _layers.append(block)
            d = block.d_output

        self.d_output = d
        self.layers = nn.ModuleList(_layers)
        if prenorm:
            if norm is None:
                self.norm = None
            elif isinstance(norm, str):
                self.norm = Normalization(
                    self.d_output, transposed=self.transposed, _name_=norm
                )
            else:
                self.norm = Normalization(
                    self.d_output, transposed=self.transposed, **norm
                )
        else:
            self.norm = nn.Identity()

    def forward(self, inputs, *args, state=None, **kwargs):
        """Inputs assumed to be (batch, sequence, dim)"""
        if self.transposed:
            inputs = rearrange(inputs, "b ... d -> b d ...")
        inputs = self.drop(inputs)

        # Track norms
        if self.track_norms:
            output_norms = [torch.mean(inputs.detach() ** 2)]

        # Apply layers
        outputs = inputs
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            outputs, state = layer(outputs, *args, state=prev_state, **kwargs)
            next_states.append(state)
            if self.track_norms:
                output_norms.append(torch.mean(outputs.detach() ** 2))
        if self.norm is not None:
            outputs = self.norm(outputs)

        if self.transposed:
            outputs = rearrange(outputs, "b d ... -> b ... d")

        if self.track_norms:
            metrics = to_dict(output_norms, recursive=False)
            self.metrics = {f"norm/{i}": v for i, v in metrics.items()}

        return outputs, next_states

    @property
    def d_state(self):
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def state_to_tensor(self):
        # Slightly hacky way to implement this in a curried manner
        # (so that the function can be extracted from an instance)
        # Somewhat more sound may be to turn this into a
        # @staticmethod and grab subclasses using hydra.utils.get_class
        def fn(state):
            x = [
                _layer.state_to_tensor(_state)
                for (_layer, _state) in zip(self.layers, state)
            ]
            x = [_x for _x in x if _x is not None]
            return torch.cat(x, dim=-1)

        return fn

    def default_state(self, *batch_shape, device=None):
        return [
            layer.default_state(*batch_shape, device=device) for layer in self.layers
        ]

    def step(self, x, state, **kwargs):
        # Apply layers
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            x, state = layer.step(x, state=prev_state, **kwargs)
            next_states.append(state)

        x = self.norm(x)
        return x, next_states
