# This code is derived from https://github.com/HazyResearch/state-spaces

import functools

from torch import nn


class SequenceModule(nn.Module):
    """Abstract sequence model class.

    All models must adhere to this interface

    A SequenceModule is generally a model that transforms an input of shape
    (n_batch, l_sequence, d_model) to (n_batch, l_sequence, d_output)

    REQUIRED methods and attributes
    forward, d_model, d_output: controls standard forward pass,
    a sequence-to-sequence transformation
    __init__ should also satisfy the following interface;
    see SequenceIdentity for an example
        def __init__(self, d_model, transposed=False, **kwargs)

    OPTIONAL methods
    default_state, step: allows stepping the model recurrently with a hidden state
    state_to_tensor, d_state: allows decoding from hidden state
    """

    @property
    def d_model(self):
        """Model dimension (generally same as input dimension).

        This attribute is required for all SequenceModule instantiations.
        It is used by the rest of the pipeline
        (e.g. model backbone, encoder) to track the internal shapes of the full model.
        """
        if getattr(self, "_d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_model")
        return self._d_model

    @d_model.setter
    def d_model(self, d):
        self._d_model = d

    @property
    def d_output(self):
        """Output dimension of model.

        This attribute is required for all SequenceModule instantiations.
        It is used by the rest of the pipeline
        (e.g. model backbone, decoder) to track the internal shapes of the full model.
        """
        if getattr(self, "_d_output", None) is None:
            raise NotImplementedError(
                "SequenceModule instantiation must specify d_output for decoder"
            )
        return self._d_output

    @d_output.setter
    def d_output(self, d):
        self._d_output = d

    def forward(self, x, state=None, **kwargs):
        """Forward pass.

        A sequence-to-sequence transformation with an optional state.

        Generally, this should map a tensor of shape
        (batch, length, self.d_model) to (batch, length, self.d_output)

        Additionally, it returns a "state" which can be any additional information
        For example, RNN and SSM layers may return their hidden state,
        while some types of transformer layers
        (e.g. Transformer-XL) may want to pass a state as well
        """
        return x, None

    @property
    def state_to_tensor(self):
        """Return a function mapping a state to a single tensor.

        This method should be implemented if one wants to use
        the hidden state insteadof the output sequence for final prediction.
        Currently only used with the StateDecoder.
        """
        return lambda _: None

    @property
    def d_state(self):
        """Return dimension of output of self.state_to_tensor."""
        return None

    def default_state(self, *batch_shape, device=None):
        """Create initial state for a batch of inputs."""
        return None

    def step(self, x, state=None, **kwargs):
        """Step the model recurrently for one step of the input sequence.

        For example, this should correspond to unrolling an RNN for one step.
        If the forward pass has signature (B, L, H1) -> (B, L, H2),
        this method should generally have signature
        (B, H1) -> (B, H2) with an optional recurrent state.
        """
        raise NotImplementedError


def TransposedModule(module):
    """Transpose module.

    Wrap a SequenceModule class to accept transposed parameter,
    handle state, absorb kwargs
    """
    # https://stackoverflow.com/a/65470430/1980685
    @functools.wraps(module, updated=())
    class TransposedModule(module):
        def __init__(self, *args, transposed=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.transposed = transposed

        def forward(self, x, state=None, **kwargs):
            if self.transposed:
                x = x.transpose(-1, -2)
            x, next_state = super().forward(x, state)  # Don't use kwarg because nn.LSTM
            next_state = None if state is None else next_state
            if self.transposed:
                x = x.transpose(-1, -2)
            return x, next_state

    # https://stackoverflow.com/questions/5352781/how-to-set-class-names-dynamically
    # TransposedModule.__name__ = module.__name__ # functools wraps is better solution
    return TransposedModule


@TransposedModule
class SequenceIdentity(SequenceModule):
    """Simple SequenceModule for testing purposes."""

    def __init__(self, d_model, dropout=0.0, **kwargs):
        """Initialize SequenceModule.

        d_model: input dimension (sometimes denoted H for hidden dimension)
        transposed: if True, inputs have axis ordering (B, H, L) instead of (B, H, L)
        """
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model

    def forward(self, x, state=None):
        """Forward pass."""
        return x, state

    def default_state(self, *batch_shape, device=None):
        """Create initial state for a batch of inputs."""
        return None

    def step(self, x, state=None, **kwargs):
        """Step the model recurrently for one step of the input sequence."""
        return x, state
