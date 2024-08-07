# This code is derived from https://github.com/HazyResearch/state-spaces

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from espnet2.asr.state_spaces.base import SequenceModule
from espnet2.asr.state_spaces.block import SequenceResidualBlock
from espnet2.asr.state_spaces.components import DropoutNd, Normalization
from espnet2.asr.state_spaces.utils import to_dict, to_list


class SequenceModel(SequenceModule):
    """
        Isotropic deep sequence model backbone, in the style of ResNets / Transformers.

    This class implements a generic (batch, length, d_input) -> (batch, length, d_output) transformation.

    Args:
        d_model (int): Dimension of the model. Used to resize input (useful for deep models with residuals).
        n_layers (int, optional): Number of layers. Defaults to 1.
        transposed (bool, optional): If True, transpose inputs so each layer receives (batch, dim, length). Defaults to False.
        dropout (float, optional): Dropout parameter applied on every residual and every layer. Defaults to 0.0.
        tie_dropout (bool, optional): If True, tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d. Defaults to False.
        prenorm (bool, optional): If True, use pre-norm instead of post-norm. Defaults to True.
        n_repeat (int, optional): Number of times each layer is repeated per stage before applying pooling. Defaults to 1.
        layer (dict or list, optional): Layer configuration. Must be specified.
        residual (dict, optional): Residual configuration.
        norm (dict or str, optional): Normalization configuration (e.g., layer vs batch).
        pool (dict, optional): Configuration for pooling layer per stage.
        track_norms (bool, optional): If True, log norms of each layer output. Defaults to True.
        dropinp (float, optional): Input dropout rate. Defaults to 0.0.
        drop_path (float, optional): Stochastic depth rate for each residual path. Defaults to 0.0.

    Attributes:
        d_output (int): Output dimension of the model.
        layers (nn.ModuleList): List of SequenceResidualBlock modules.
        norm (nn.Module): Normalization layer (if prenorm is True) or nn.Identity.

    Examples:
        >>> model = SequenceModel(d_model=256, n_layers=4, dropout=0.1)
        >>> inputs = torch.randn(32, 100, 256)  # (batch, length, d_input)
        >>> outputs, _ = model(inputs)
        >>> outputs.shape
        torch.Size([32, 100, 256])

    Note:
        The model can handle both transposed and non-transposed inputs, depending on the 'transposed' parameter.
        It also supports various configurations for normalization, residual connections, and pooling.
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
        """
                Forward pass of the SequenceModel.

        This method processes the input through all layers of the model, applying
        dropout, normalization, and tracking norms if specified.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch, sequence, dim) or
                (batch, dim, sequence) if self.transposed is True.
            *args: Variable length argument list.
            state (list, optional): List of previous states for each layer. If None,
                default states will be used. Defaults to None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: A tuple containing:
                - outputs (torch.Tensor): The processed output tensor of shape
                  (batch, sequence, d_output) or (batch, d_output, sequence) if
                  self.transposed is True.
                - next_states (list): List of updated states for each layer.

        Examples:
            >>> model = SequenceModel(d_model=256, n_layers=4)
            >>> inputs = torch.randn(32, 100, 256)
            >>> outputs, states = model.forward(inputs)
            >>> outputs.shape
            torch.Size([32, 100, 256])
            >>> len(states)
            4

        Note:
            If self.track_norms is True, the method will update self.metrics with
            the mean squared values of the outputs at each layer.
        """
        # Inputs assumed to be (batch, sequence, dim)
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
        """
                Total dimension of the state across all layers.

        This property calculates and returns the sum of state dimensions for all layers
        in the model that have a state.

        Returns:
            int: The total dimension of the state across all layers. If a layer doesn't
            have a state (i.e., its d_state is None), it's not included in the sum.

        Examples:
            >>> model = SequenceModel(d_model=256, n_layers=4)
            >>> model.d_state
            1024  # Assuming each layer has a state dimension of 256

        Note:
            This property is useful for understanding the total state size of the model,
            which can be important for memory considerations or when initializing or
            processing the entire state of the model at once.
        """
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def state_to_tensor(self):
        """
                Property that returns a function to convert model state to a tensor.

        This property provides a function that concatenates the tensor representations
        of states from all layers into a single tensor.

        Returns:
            function: A function that takes a state (list of layer states) as input
            and returns a concatenated tensor of all non-None states.

        Examples:
            >>> model = SequenceModel(d_model=256, n_layers=4)
            >>> state = model.default_state(batch_size=32)
            >>> state_tensor = model.state_to_tensor(state)
            >>> state_tensor.shape
            torch.Size([32, 1024])  # Assuming each layer has a state dimension of 256

        Note:
            This property uses a closure to create a function that has access to
            the model's layers. The returned function is "curried" in the sense that
            it's partially applied to the model's layers and can be called later
            with a state argument.

            The function skips any None states, only concatenating tensor states.
        """

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
        """
                Generate the default initial state for the model.

        This method creates a list of default states for each layer in the model.

        Args:
            *batch_shape (tuple): Variable length argument list for the batch dimensions.
            device (torch.device, optional): The device on which to create the state tensors.
                If None, uses the default device. Defaults to None.

        Returns:
            list: A list of default states for each layer in the model. The structure and
            content of each state depends on the specific implementation of each layer.

        Examples:
            >>> model = SequenceModel(d_model=256, n_layers=4)
            >>> state = model.default_state(32)  # for a batch size of 32
            >>> len(state)
            4  # one state per layer
            >>> state = model.default_state(32, device=torch.device('cuda'))  # on GPU

        Note:
            The shape and content of the default state for each layer may vary depending
            on the layer type and configuration. Some layers might return None if they
            don't maintain a state.

            This method is particularly useful for initializing the model's state at the
            beginning of a sequence or when no previous state is available.
        """
        return [
            layer.default_state(*batch_shape, device=device) for layer in self.layers
        ]

    def step(self, x, state, **kwargs):
        """
                Perform a single step forward pass of the model.

        This method applies the model to a single step of input, updating the state
        for each layer in the process.

        Args:
            x (torch.Tensor): Input tensor for a single step. The shape should be
                compatible with the model's input requirements for a single time step.
            state (list, optional): List of previous states for each layer. If None,
                default states will be used. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to each layer's step method.

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): The output tensor after processing through all layers.
                - next_states (list): List of updated states for each layer.

        Examples:
            >>> model = SequenceModel(d_model=256, n_layers=4)
            >>> x = torch.randn(32, 256)  # (batch_size, d_model)
            >>> state = model.default_state(32)
            >>> output, new_state = model.step(x, state)
            >>> output.shape
            torch.Size([32, 256])
            >>> len(new_state)
            4

        Note:
            This method is particularly useful for processing sequences one step at a time,
            which can be more memory-efficient for very long sequences or in scenarios
            where future inputs are not available, such as in real-time or streaming applications.

            The final normalization layer is applied to the output before returning.
        """
        # Apply layers
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            x, state = layer.step(x, state=prev_state, **kwargs)
            next_states.append(state)

        x = self.norm(x)
        return x, next_states
