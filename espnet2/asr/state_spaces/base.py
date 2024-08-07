# This code is derived from https://github.com/HazyResearch/state-spaces

import functools

from torch import nn


class SequenceModule(nn.Module):
    """
        Abstract base class for sequence models.

    This class defines the interface for sequence models that transform input tensors
    of shape (n_batch, l_sequence, d_model) to (n_batch, l_sequence, d_output).

    All models inheriting from this class must implement the required methods and
    attributes. Optional methods provide additional functionality for recurrent
    processing and state management.

    Attributes:
        d_model (int): Model dimension (generally same as input dimension).
        d_output (int): Output dimension of the model.

    Args:
        d_model (int): The input dimension of the model.
        transposed (bool, optional): If True, expects input in (n_batch, d_model, l_sequence)
            format. Defaults to False.

    Required Methods:
        forward(self, x, state=None, **kwargs): Performs the forward pass of the model.
        __init__(self, d_model, transposed=False, **kwargs): Initializes the model.

    Optional Methods:
        default_state(self, *batch_shape, device=None): Creates initial state for a batch.
        step(self, x, state=None, **kwargs): Processes one step of the input sequence.
        state_to_tensor(self): Returns a function to map state to a single tensor.

    Properties:
        d_state (int): Dimension of the output of self.state_to_tensor.

    Example:
        class MySequenceModel(SequenceModule):
            def __init__(self, d_model, d_output):
                super().__init__()
                self.d_model = d_model
                self.d_output = d_output
                self.linear = nn.Linear(d_model, d_output)

            def forward(self, x, state=None):
                return self.linear(x), None

        model = MySequenceModel(d_model=64, d_output=32)
        x = torch.randn(16, 100, 64)  # (batch, sequence, d_model)
        output, _ = model(x)  # output shape: (16, 100, 32)

    Note:
        Subclasses must set self._d_model and self._d_output in their __init__ method.
    """

    @property
    def d_model(self):
        """
                Model dimension (generally same as input dimension).

        This property is required for all SequenceModule instantiations. It is used by
        the rest of the pipeline (e.g., model backbone, encoder) to track the internal
        shapes of the full model.

        Returns:
            int: The model dimension.

        Raises:
            NotImplementedError: If the SequenceModule instantiation has not set d_model.

        Note:
            Subclasses must set self._d_model in their __init__ method.

        Example:
            class MySequenceModel(SequenceModule):
                def __init__(self, d_model):
                    super().__init__()
                    self._d_model = d_model  # Set the internal attribute

                # ... other methods ...

            model = MySequenceModel(64)
            print(model.d_model)  # Output: 64
        """
        if getattr(self, "_d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_model")
        return self._d_model

    @d_model.setter
    def d_model(self, d):
        """
                Model dimension (generally same as input dimension).

        This property is required for all SequenceModule instantiations. It is used by
        the rest of the pipeline (e.g., model backbone, encoder) to track the internal
        shapes of the full model.

        Returns:
            int: The model dimension.

        Raises:
            NotImplementedError: If the SequenceModule instantiation has not set d_model.

        Note:
            Subclasses must set self._d_model in their __init__ method.

        Example:
            class MySequenceModel(SequenceModule):
                def __init__(self, d_model):
                    super().__init__()
                    self._d_model = d_model  # Set the internal attribute

                # ... other methods ...

            model = MySequenceModel(64)
            print(model.d_model)  # Output: 64
        """
        self._d_model = d

    @property
    def d_output(self):
        """
                Output dimension of the model.

        This property is required for all SequenceModule instantiations. It is used by
        the rest of the pipeline (e.g., model backbone, decoder) to track the internal
        shapes of the full model.

        Returns:
            int: The output dimension of the model.

        Raises:
            NotImplementedError: If the SequenceModule instantiation has not specified
                d_output for the decoder.

        Note:
            Subclasses must set self._d_output in their __init__ method.

        Example:
            class MySequenceModel(SequenceModule):
                def __init__(self, d_model, d_output):
                    super().__init__()
                    self._d_model = d_model
                    self._d_output = d_output  # Set the internal attribute

                # ... other methods ...

            model = MySequenceModel(d_model=64, d_output=32)
            print(model.d_output)  # Output: 32
        """
        if getattr(self, "_d_output", None) is None:
            raise NotImplementedError(
                "SequenceModule instantiation must specify d_output for decoder"
            )
        return self._d_output

    @d_output.setter
    def d_output(self, d):
        """
                Output dimension of the model.

        This property is required for all SequenceModule instantiations. It is used by
        the rest of the pipeline (e.g., model backbone, decoder) to track the internal
        shapes of the full model.

        Returns:
            int: The output dimension of the model.

        Raises:
            NotImplementedError: If the SequenceModule instantiation has not specified
                d_output for the decoder.

        Note:
            Subclasses must set self._d_output in their __init__ method.

        Example:
            class MySequenceModel(SequenceModule):
                def __init__(self, d_model, d_output):
                    super().__init__()
                    self._d_model = d_model
                    self._d_output = d_output  # Set the internal attribute

                # ... other methods ...

            model = MySequenceModel(d_model=64, d_output=32)
            print(model.d_output)  # Output: 32
        """
        self._d_output = d

    def forward(self, x, state=None, **kwargs):
        """
                Perform the forward pass of the sequence model.

        This method implements a sequence-to-sequence transformation with an optional state.
        It should map a tensor of shape (batch, length, self.d_model) to
        (batch, length, self.d_output).

        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, self.d_model).
            state (Any, optional): Initial state for the forward pass. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor of shape (batch, length, self.d_output).
                - Any: Updated state. This can be any additional information, such as
                  the hidden state for RNN and SSM layers, or state information for
                  certain types of transformer layers (e.g., Transformer-XL).

        Example:
            class MySequenceModel(SequenceModule):
                def __init__(self, d_model, d_output):
                    super().__init__()
                    self.d_model = d_model
                    self.d_output = d_output
                    self.linear = nn.Linear(d_model, d_output)

                def forward(self, x, state=None):
                    output = self.linear(x)
                    return output, None

            model = MySequenceModel(d_model=64, d_output=32)
            x = torch.randn(16, 100, 64)  # (batch, length, d_model)
            output, new_state = model(x)
            print(output.shape)  # torch.Size([16, 100, 32])
        """
        return x, None

    @property
    def state_to_tensor(self):
        """
                Function to map a state to a single tensor.

        This property should return a function that converts the model's state into a
        single tensor representation. It is primarily used when the hidden state, rather
        than the output sequence, is needed for final prediction. This method is
        currently only used in conjunction with the StateDecoder.

        Returns:
            Callable: A function that takes a state as input and returns a tensor
                representation of that state. If not implemented, returns a lambda
                function that always returns None.

        Note:
            Subclasses should override this property if they want to provide a custom
            state-to-tensor conversion.

        Example:
            class MyRNNModel(SequenceModule):
                def __init__(self, d_model, d_hidden):
                    super().__init__()
                    self.rnn = nn.RNN(d_model, d_hidden)
                    self._d_state = d_hidden

                @property
                def state_to_tensor(self):
                    return lambda state: state[0].squeeze(0)

                # ... other methods ...

            model = MyRNNModel(d_model=64, d_hidden=32)
            x = torch.randn(16, 100, 64)  # (batch, length, d_model)
            _, final_state = model(x)
            state_tensor = model.state_to_tensor(final_state)
            print(state_tensor.shape)  # torch.Size([16, 32])
        """
        return lambda _: None

    @property
    def d_state(self):
        """
                Dimension of the output of self.state_to_tensor.

        This property returns the dimension of the tensor produced by the state_to_tensor
        method. It is useful for determining the size of the state representation,
        particularly when using the state for downstream tasks or in the StateDecoder.

        Returns:
            int or None: The dimension of the state tensor. Returns None if not implemented
            or if the state does not have a fixed dimension.

        Note:
            Subclasses should override this property if they implement a custom
            state_to_tensor method with a known output dimension.

        Example:
            class MyLSTMModel(SequenceModule):
                def __init__(self, d_model, d_hidden):
                    super().__init__()
                    self.lstm = nn.LSTM(d_model, d_hidden)
                    self._d_state = d_hidden

                @property
                def d_state(self):
                    return self._d_state

                @property
                def state_to_tensor(self):
                    return lambda state: state[0].view(state[0].size(1), -1)

                # ... other methods ...

            model = MyLSTMModel(d_model=64, d_hidden=32)
            print(model.d_state)  # Output: 32
        """
        return None

    def default_state(self, *batch_shape, device=None):
        """
                Create initial state for a batch of inputs.

        This method generates the default initial state for the sequence model. It is
        particularly useful for models that maintain internal states, such as RNNs or
        state space models.

        Args:
            *batch_shape: Variable length argument list for the batch dimensions.
            device (torch.device, optional): The device on which to create the state.
                Defaults to None, which means the state will be created on the default device.

        Returns:
            Any: The initial state for the model. The exact type and shape depend on the
            specific implementation. Returns None by default if not implemented.

        Example:
            class MyGRUModel(SequenceModule):
                def __init__(self, d_model, d_hidden):
                    super().__init__()
                    self.gru = nn.GRU(d_model, d_hidden)
                    self.d_hidden = d_hidden

                def default_state(self, *batch_shape, device=None):
                    return torch.zeros(*batch_shape, self.d_hidden, device=device)

                # ... other methods ...

            model = MyGRUModel(d_model=64, d_hidden=32)
            initial_state = model.default_state(16, device=torch.device('cuda'))
            print(initial_state.shape)  # torch.Size([16, 32])

        Note:
            Subclasses should override this method if they require a non-None initial state.
            The method should be able to handle variable batch dimensions.
        """
        return None

    def step(self, x, state=None, **kwargs):
        """
                Process one step of the input sequence.

        This method steps the model recurrently for a single step of the input sequence.
        It is particularly useful for models that need to process sequences step-by-step,
        such as in autoregressive generation or online inference scenarios.

        Args:
            x (torch.Tensor): Input tensor for a single step. If the forward pass has
                signature (B, L, H1) -> (B, L, H2), this method's input should generally
                have shape (B, H1), where B is the batch size and H1 is the input dimension.
            state (Any, optional): The current state of the model. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor for the single step, typically of shape (B, H2),
                  where H2 is the output dimension.
                - Any: The updated state after processing this step.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            class MyRNNModel(SequenceModule):
                def __init__(self, d_model, d_hidden):
                    super().__init__()
                    self.rnn_cell = nn.RNNCell(d_model, d_hidden)
                    self.d_model = d_model
                    self.d_output = d_hidden

                def step(self, x, state=None):
                    if state is None:
                        state = torch.zeros(x.size(0), self.d_output, device=x.device)
                    output = self.rnn_cell(x, state)
                    return output, output

                # ... other methods ...

            model = MyRNNModel(d_model=64, d_hidden=32)
            x = torch.randn(16, 64)  # (batch_size, d_model)
            output, new_state = model.step(x)
            print(output.shape)  # torch.Size([16, 32])

        Note:
            This method should be implemented by subclasses that support step-by-step
            processing. The exact signature may vary depending on the specific model architecture.
        """
        raise NotImplementedError


def TransposedModule(module):
    """
        Decorator to transpose input and output of a SequenceModule.

    This decorator wraps a SequenceModule class to handle transposed input and output,
    manage state, and absorb additional keyword arguments. It allows the module to
    operate on either (B, L, H) or (B, H, L) shaped inputs, where B is batch size,
    L is sequence length, and H is hidden dimension.

    Attributes:
        transposed (bool): If True, the input and output tensors are transposed.

    Args:
        module (type): The SequenceModule class to be wrapped.

    Returns:
        type: A new class that wraps the input module with transposition functionality.

    Example:
        @TransposedModule
        class MySequenceModule(SequenceModule):
            def __init__(self, d_model):
                super().__init__()
                self.d_model = d_model
                self.d_output = d_model

            def forward(self, x, state=None):
                # Process x
                return x, state

        # Now MySequenceModule can handle both (B, L, H) and (B, H, L) inputs
        model = MySequenceModule(d_model=64, transposed=True)
        x = torch.randn(32, 64, 100)  # (B, H, L)
        output, _ = model(x)  # output will be (32, 64, 100)

    Note:
        The wrapped module's forward method should accept 'state' as an argument
        and return both the processed tensor and the next state.
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
    """
        A simple identity SequenceModule for testing purposes.

    This class implements a basic SequenceModule that passes the input through
    unchanged. It is primarily used for testing and as a minimal example of
    a SequenceModule implementation.

    The SequenceIdentity class is wrapped with the @TransposedModule decorator,
    allowing it to handle both standard and transposed input formats.

    Attributes:
        d_model (int): The input and output dimension of the model.
        d_output (int): Alias for d_model, as this module doesn't change dimensions.

    Args:
        d_model (int): The dimension of the input and output.
        dropout (float, optional): Dropout rate (unused in this implementation).
            Defaults to 0.0.
        **kwargs: Additional keyword arguments (unused).

    Example:
        model = SequenceIdentity(d_model=64)
        x = torch.randn(32, 100, 64)  # (batch_size, sequence_length, d_model)
        output, _ = model(x)
        assert torch.allclose(x, output)  # True, as this is an identity module

        # With transposed input
        model_transposed = SequenceIdentity(d_model=64, transposed=True)
        x_transposed = torch.randn(32, 64, 100)  # (batch_size, d_model, sequence_length)
        output_transposed, _ = model_transposed(x_transposed)
        assert torch.allclose(x_transposed, output_transposed)  # True

    Note:
        This class is wrapped with @TransposedModule, which allows it to handle
        both (B, L, H) and (B, H, L) input formats based on the 'transposed' parameter.
    """

    def __init__(self, d_model, dropout=0.0, **kwargs):
        """Initialize SequenceModule.

        d_model: input dimension (sometimes denoted H for hidden dimension)
        transposed: if True, inputs have axis ordering (B, H, L) instead of (B, H, L)
        """
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model

    def forward(self, x, state=None):
        """
                Perform the forward pass of the SequenceIdentity module.

        This method implements the identity operation, returning the input unchanged.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, d_model) if not transposed,
                or (batch, d_model, length) if transposed.
            state (Any, optional): Unused in this implementation. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The input tensor x, unchanged.
                - None: As this module doesn't maintain state.

        Example:
            model = SequenceIdentity(d_model=64)
            x = torch.randn(32, 100, 64)  # (batch_size, sequence_length, d_model)
            output, _ = model(x)
            assert torch.allclose(x, output)  # True, as this is an identity operation

        Note:
            If the module was initialized with transposed=True, the method expects
            and returns tensors in the shape (batch, d_model, length).
        """
        return x, state

    def default_state(self, *batch_shape, device=None):
        """
                Create initial state for a batch of inputs.

        This method returns None as the SequenceIdentity module does not maintain any state.

        Args:
            *batch_shape: Variable length argument list for the batch dimensions (unused).
            device (torch.device, optional): The device on which to create the state (unused).
                Defaults to None.

        Returns:
            None: As SequenceIdentity doesn't use any state.

        Example:
            model = SequenceIdentity(d_model=64)
            initial_state = model.default_state(32)  # batch size of 32
            assert initial_state is None  # True

        Note:
            This method is implemented to conform to the SequenceModule interface,
            but it doesn't perform any meaningful operation for SequenceIdentity.
        """
        return None

    def step(self, x, state=None, **kwargs):
        """
                Process one step of the input sequence.

        This method implements the identity operation for a single step, returning the input unchanged.

        Args:
            x (torch.Tensor): Input tensor for a single step, typically of shape (batch, d_model).
            state (Any, optional): Unused in this implementation. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The input tensor x, unchanged.
                - None: As this module doesn't maintain state.

        Example:
            model = SequenceIdentity(d_model=64)
            x = torch.randn(32, 64)  # (batch_size, d_model)
            output, _ = model.step(x)
            assert torch.allclose(x, output)  # True, as this is an identity operation

        Note:
            This method is implemented to conform to the SequenceModule interface.
            For SequenceIdentity, it performs the same operation as the forward method,
            but for a single step of the sequence.
        """
        return x, state
