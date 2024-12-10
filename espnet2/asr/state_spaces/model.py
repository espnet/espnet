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
    Isotropic deep sequence model backbone, inspired by ResNets and Transformers.

    The SequenceModel class implements a generic transformation from
    (batch, length, d_input) to (batch, length, d_output). This model can be
    configured with various parameters to adjust its architecture and behavior.

    Attributes:
        d_model (int): Dimensionality of the input features.
        transposed (bool): If True, input tensors are transposed.
        track_norms (bool): If True, logs the norms of each layer output.
        drop (nn.Module): Dropout layer applied to the inputs.
        layers (nn.ModuleList): List of sequential residual blocks.
        norm (nn.Module): Normalization layer applied at the end, if specified.
        d_output (int): Dimensionality of the output features.

    Args:
        d_model (int): The dimensionality of the model input.
        n_layers (int): Number of layers in the model. Default is 1.
        transposed (bool): If True, transposes the input shape. Default is False.
        dropout (float): Dropout rate applied on each residual connection. Default is 0.0.
        tie_dropout (bool): If True, ties dropout mask across sequence like
            nn.Dropout1d/nn.Dropout2d. Default is False.
        prenorm (bool): If True, applies normalization before the layer. Default is True.
        n_repeat (int): Number of times each layer is repeated before pooling. Default is 1.
        layer (dict or list): Configuration for the layers. Must be specified.
        residual (dict): Configuration for the residual connections.
        norm (dict or str): Normalization configuration (e.g. 'layer', 'batch').
        pool (dict): Configuration for pooling layer per stage.
        track_norms (bool): If True, tracks and logs the norms of each layer output. Default is True.
        dropinp (float): Dropout rate applied to inputs. Default is 0.0.
        drop_path (float): Stochastic depth for each residual path. Default is 0.0.

    Returns:
        tuple: A tuple containing:
            - outputs (torch.Tensor): The output tensor of shape
              (batch, length, d_output).
            - next_states (list): The updated states after processing through layers.

    Raises:
        ValueError: If the layer configuration is invalid.

    Examples:
        >>> model = SequenceModel(d_model=128, n_layers=4, layer=[{'type': 'conv'}])
        >>> inputs = torch.randn(32, 10, 128)  # (batch, length, d_input)
        >>> outputs, states = model(inputs)
        >>> print(outputs.shape)
        torch.Size([32, 10, d_output])  # d_output depends on layer configuration

    Note:
        This model can be used for various sequence modeling tasks such as
        automatic speech recognition (ASR) and other sequence-based applications.

    Todo:
        - Implement additional layer types and configurations.
        - Optimize the forward pass for better performance on large sequences.
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
        Forward pass for the SequenceModel, which processes the input tensor
        through the defined layers and applies normalization if specified.

        This method assumes that the input tensor is shaped as
        (batch, sequence, dim) and applies dropout, layers, and normalization
        sequentially.

        Args:
            inputs (torch.Tensor): The input tensor of shape
                (batch, sequence, dim).
            *args: Additional positional arguments passed to each layer.
            state (list, optional): A list of previous states for each layer.
                If None, initializes to a list of None.
            **kwargs: Additional keyword arguments passed to each layer.

        Returns:
            tuple: A tuple containing:
                - outputs (torch.Tensor): The output tensor after processing,
                  shaped as (batch, sequence, d_output).
                - next_states (list): A list of states for each layer after
                  processing.

        Raises:
            ValueError: If the input tensor does not match the expected
            shape.

        Examples:
            >>> model = SequenceModel(d_model=128, n_layers=3)
            >>> inputs = torch.randn(32, 10, 128)  # (batch, sequence, dim)
            >>> outputs, states = model(inputs)

        Note:
            The method tracks the norms of outputs at each layer if
            `track_norms` is set to True, which can be accessed via
            the `metrics` attribute after the forward pass.
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
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def state_to_tensor(self):
        """
        Convert the state of each layer into a tensor representation.

        This method iterates through the layers of the sequence model, calling
        each layer's `state_to_tensor` method on the corresponding state. It
        concatenates the resulting tensors along the last dimension to produce a
        single tensor that represents the entire state of the model.

        Args:
            state (list): A list of states, one for each layer in the model.

        Returns:
            torch.Tensor: A tensor containing the concatenated states of all layers.

        Examples:
            >>> model = SequenceModel(d_model=128, n_layers=2)
            >>> states = model.default_state(batch_shape=(10,), device='cpu')
            >>> tensor_representation = model.state_to_tensor(states)
            >>> print(tensor_representation.shape)
            torch.Size([10, d_output])  # where d_output is the concatenated dimension

        Note:
            This method assumes that each layer's `state_to_tensor` method is
            implemented and returns a tensor. If any layer returns `None`, it
            will be excluded from the concatenation.

        Todo:
            Consider refactoring this method to be a static method if it
            does not rely on instance-specific data.
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
        Generate the default state for each layer in the sequence model.

        This method creates an initial state for each layer based on the specified
        batch shape and device. The default state can be used as a starting point
        for processing inputs through the model.

        Args:
            *batch_shape: Variable length argument for the shape of the batch.
                        This should typically represent the dimensions of the
                        input sequence excluding the last dimension, which is
                        the feature dimension.
            device (torch.device, optional): The device on which to create the
                                            state tensors. If not specified,
                                            the default device will be used.

        Returns:
            list: A list containing the default state tensors for each layer.
                Each tensor's shape is determined by the layer's internal
                state configuration and the provided batch shape.

        Examples:
            >>> model = SequenceModel(d_model=128, n_layers=3)
            >>> default_states = model.default_state(10, 20)  # For a batch size of 10
            >>> print([state.shape for state in default_states])
            [torch.Size([10, ...]), torch.Size([10, ...]), torch.Size([10, ...])]

        Note:
            The shapes of the returned state tensors depend on the individual
            layer configurations and the specified batch shape. Each layer may
            have a different state shape based on its design.

        Raises:
            ValueError: If the batch shape is invalid or incompatible with the
                        model's architecture.
        """
        return [
            layer.default_state(*batch_shape, device=device) for layer in self.layers
        ]

    def step(self, x, state, **kwargs):
        """
        Processes a single time step of input through the model layers.

        This method applies each layer of the sequence model to the input
        tensor `x` and updates the hidden states. It is typically used in
        scenarios where the model needs to process input sequentially,
        such as in recurrent architectures or during inference.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_input).
            state (list): List of previous states for each layer, or None
                if no state is to be used.
            **kwargs: Additional keyword arguments passed to each layer's
                step method.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after processing through
                  all layers of shape (batch_size, d_output).
                - list: Updated list of states for each layer.

        Examples:
            >>> model = SequenceModel(d_model=64, n_layers=2)
            >>> input_tensor = torch.randn(32, 10, 64)  # (batch_size, seq_len, d_input)
            >>> initial_state = model.default_state(32)  # Default state for the batch
            >>> output, next_state = model.step(input_tensor, initial_state)

        Note:
            This method is designed to work with the assumption that
            the input `x` is formatted as (batch_size, d_input).
        """
        # Apply layers
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            x, state = layer.step(x, state=prev_state, **kwargs)
            next_states.append(state)

        x = self.norm(x)
        return x, next_states
