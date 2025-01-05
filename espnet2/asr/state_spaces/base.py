# This code is derived from https://github.com/HazyResearch/state-spaces

import functools

from torch import nn


class SequenceModule(nn.Module):
    """
    SequenceModule is an abstract class that defines the interface for sequence
    models in a neural network framework. It transforms an input tensor of shape
    (n_batch, l_sequence, d_model) into an output tensor of shape
    (n_batch, l_sequence, d_output).

    This class requires implementations of the `forward` method and the
    `d_model` and `d_output` attributes to facilitate a standard sequence-to-
    sequence transformation. Additionally, it provides optional methods for
    state management, enabling recurrent processing and state decoding.

    Attributes:
        d_model (int): The model dimension, generally the same as the input
            dimension. This attribute must be set during initialization.
        d_output (int): The output dimension of the model. This attribute must
            also be set during initialization.

    Methods:
        forward(x, state=None, **kwargs): Performs a forward pass through the
            model, mapping input tensors to output tensors.
        default_state(*batch_shape, device=None): Creates an initial state for
            a batch of inputs.
        step(x, state=None, **kwargs): Processes one step of the input sequence
            recurrently.
        state_to_tensor: A property that returns a function to map the hidden
            state to a tensor.
        d_state: A property that returns the dimension of the output of
            `state_to_tensor`.

    Raises:
        NotImplementedError: If the required attributes or methods are not
            implemented in a subclass.

    Examples:
        To create a custom sequence model, subclass `SequenceModule` and
        implement the required methods:

        ```python
        class MySequenceModel(SequenceModule):
            def __init__(self, d_model, d_output):
                super().__init__()
                self.d_model = d_model
                self.d_output = d_output

            def forward(self, x, state=None):
                # Implement the transformation logic here
                return transformed_x, new_state
        ```

        When using the model:

        ```python
        model = MySequenceModel(d_model=128, d_output=64)
        output, state = model.forward(input_tensor)
        ```

    Note:
        This class is part of the ESPnet2 ASR framework and serves as a
        foundational building block for various sequence models.
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
        """
        Abstract sequence model class.

        All models must adhere to this interface.

        A SequenceModule is generally a model that transforms an input of shape
        (n_batch, l_sequence, d_model) to (n_batch, l_sequence, d_output).

        REQUIRED methods and attributes:
            forward, d_model, d_output: controls standard forward pass,
            a sequence-to-sequence transformation.
            __init__ should also satisfy the following interface;
            see SequenceIdentity for an example:
                def __init__(self, d_model, transposed=False, **kwargs)

        OPTIONAL methods:
            default_state, step: allows stepping the model recurrently with a hidden state.
            state_to_tensor, d_state: allows decoding from hidden state.

        Attributes:
            d_model (int): Model dimension, generally the same as input dimension.
            d_output (int): Output dimension of the model.

        Examples:
            >>> model = MySequenceModel(d_model=128)
            >>> input_tensor = torch.randn(32, 10, 128)  # (batch, length, features)
            >>> output, state = model(input_tensor)

        Raises:
            NotImplementedError: If d_model or d_output is not set during instantiation.
        """
        self._d_model = d

    @property
    def d_output(self):
        """
            Output dimension of model.

        This attribute is required for all instances of SequenceModule.
        It is used by the rest of the pipeline (e.g., model backbone, decoder)
        to track the internal shapes of the full model. The dimension must be
        specified during the instantiation of the model. If not set, a
        NotImplementedError will be raised.

        Returns:
            int: The output dimension of the model.

        Raises:
            NotImplementedError: If d_output is not specified during
            instantiation.

        Examples:
            >>> model = SomeSequenceModel(d_model=128, d_output=64)
            >>> model.d_output
            64

            >>> model = SomeSequenceModel(d_model=128)  # d_output not set
            >>> model.d_output  # Raises NotImplementedError
        """
        if getattr(self, "_d_output", None) is None:
            raise NotImplementedError(
                "SequenceModule instantiation must specify d_output for decoder"
            )
        return self._d_output

    @d_output.setter
    def d_output(self, d):
        """
        Output dimension of model.

        This attribute is required for all SequenceModule instantiations.
        It is used by the rest of the pipeline (e.g. model backbone, decoder)
        to track the internal shapes of the full model.

        Raises:
            NotImplementedError: If the output dimension has not been set.

        Examples:
            >>> model = SequenceIdentity(d_model=128)
            >>> model.d_output = 64
            >>> print(model.d_output)
            64
        """
        self._d_output = d

    def forward(self, x, state=None, **kwargs):
        """
        Forward pass.

        A sequence-to-sequence transformation with an optional state.

        This method takes an input tensor of shape
        (batch, length, self.d_model) and transforms it to
        (batch, length, self.d_output). The function also returns a "state"
        which can contain additional information, such as hidden states for
        RNN and SSM layers. Some transformer layers (e.g., Transformer-XL)
        may also utilize this state.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, d_model).
            state (optional): The initial state for recurrent processing,
                which may be `None` if not applicable.
            **kwargs: Additional keyword arguments that may be relevant for
                specific implementations.

        Returns:
            Tuple[torch.Tensor, Optional]: A tuple containing:
                - Output tensor of shape (batch, length, d_output).
                - Updated state (if applicable), which can be `None`.

        Examples:
            >>> model = SequenceIdentity(d_model=128)
            >>> input_tensor = torch.randn(32, 10, 128)  # (batch, length, d_model)
            >>> output, state = model.forward(input_tensor)
            >>> print(output.shape)  # Should output: torch.Size([32, 10, 128])

        Note:
            Implementations of this method should ensure that the output
            shape aligns with the specified d_output attribute.

        Raises:
            NotImplementedError: If the method is not implemented in a
                derived class.
        """
        return x, None

    @property
    def state_to_tensor(self):
        """
            @property
        def state_to_tensor(self):
        """
        return lambda _: None

    @property
    def d_state(self):
        """Return dimension of output of self.state_to_tensor."""
        return None

    def default_state(self, *batch_shape, device=None):
        """
        Create initial state for a batch of inputs.

        This method is intended to be overridden by subclasses to provide an
        appropriate initial state based on the input batch shape and device.
        The default implementation returns None, indicating that no initial
        state is used.

        Args:
            *batch_shape: Variable-length argument list representing the shape
                of the batch. This can be used to determine the size of the
                initial state.
            device (torch.device, optional): The device on which to create the
                initial state. If not specified, the default device will be
                used.

        Returns:
            Initial state tensor for the given batch shape, or None if no state
            is required.

        Examples:
            >>> module = SequenceIdentity(d_model=128)
            >>> initial_state = module.default_state(32, 10)  # Batch of 32, seq len 10
            >>> print(initial_state)  # Should output: None

        Note:
            Subclasses that require a state should implement this method to
            return a valid tensor based on the specified batch shape and device.
        """
        return None

    def step(self, x, state=None, **kwargs):
        """
        Step the model recurrently for one step of the input sequence.

        This method processes a single input step and updates the model's state.
        It is typically used in recurrent architectures, such as RNNs, to
        compute the next output based on the current input and the previous
        state.

        The method generally has the following signature:
        (B, H1) -> (B, H2), where:
            - B is the batch size
            - H1 is the dimension of the input
            - H2 is the dimension of the output

        Args:
            x (torch.Tensor): Input tensor of shape (B, H1), where B is the
                batch size and H1 is the input dimension.
            state (optional): The previous hidden state of the model, which
                can be used for recurrent processing.
            **kwargs: Additional keyword arguments that may be used by
                specific implementations.

        Returns:
            torch.Tensor: The output tensor of shape (B, H2), where H2 is
                the output dimension of the model.
            Optional: Updated state after processing the input.

        Raises:
            NotImplementedError: If the method is not implemented in a
                subclass.

        Examples:
            >>> model = MyRNNModel(d_model=128)
            >>> input_tensor = torch.randn(32, 128)  # Batch size of 32
            >>> output, new_state = model.step(input_tensor, state=prev_state)

        Note:
            This method must be overridden in subclasses of SequenceModule
            to provide specific recurrent behavior.
        """
        raise NotImplementedError


def TransposedModule(module):
    """
    Transposed module.

    This function serves as a decorator that wraps a `SequenceModule` class to
    allow it to accept a `transposed` parameter. When `transposed` is set to
    True, the input tensor's dimensions are transposed before and after the
    forward pass, enabling compatibility with different input shapes.

    Attributes:
        transposed (bool): Indicates whether the input should be transposed.

    Args:
        module (type): A subclass of `SequenceModule` that will be wrapped.

    Returns:
        type: A subclass of `module` with added transposition capabilities.

    Examples:
        >>> @TransposedModule
        >>> class MyModule(SequenceModule):
        >>>     def __init__(self, d_model):
        >>>         super().__init__()
        >>>         self.d_model = d_model
        >>>         self.d_output = d_model
        >>>
        >>> my_module = MyModule(d_model=128, transposed=True)
        >>> input_tensor = torch.randn(32, 128, 10)  # Shape (n_batch, d_model, l_sequence)
        >>> output, state = my_module(input_tensor)

    Note:
        This decorator modifies the behavior of the `forward` method to handle
        the transposed state appropriately.

    Todo:
        - Ensure compatibility with other sequence models that may require
        additional handling for their states.
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
    Class SequenceIdentity

    A simple implementation of the SequenceModule designed for testing purposes.
    This class acts as a direct identity mapping for input sequences, transforming
    input tensors of shape (batch, length, d_model) to output tensors of the same
    shape. It inherits from SequenceModule and fulfills the required interface for
    sequence models.

    Attributes:
        d_model (int): The input and output dimension of the model.
        d_output (int): The output dimension, which is equal to d_model.

    Args:
        d_model (int): Input dimension, which is also the hidden dimension.
        dropout (float): Dropout rate for regularization (not utilized in this
            implementation).
        **kwargs: Additional keyword arguments passed to the parent class.

    Returns:
        tuple: A tuple containing the output tensor and the state.

    Methods:
        forward(x, state=None): Performs the forward pass, returning the input as
            output.
        default_state(*batch_shape, device=None): Creates an initial state for a
            batch of inputs.
        step(x, state=None, **kwargs): Steps the model recurrently for one step of
            the input sequence.

    Examples:
        >>> model = SequenceIdentity(d_model=128)
        >>> input_tensor = torch.randn(10, 5, 128)  # (batch_size, seq_len, d_model)
        >>> output, state = model(input_tensor)
        >>> assert output.shape == input_tensor.shape  # Output shape matches input shape

    Note:
        This module does not implement any form of learning or state management
        beyond passing the input through unchanged. It serves primarily for testing
        and benchmarking other sequence models.

    Todo:
        - Consider implementing dropout or other regularization techniques in the
        future if needed for testing purposes.
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
        Forward pass.

        This method performs a sequence-to-sequence transformation on the input
        tensor. It maps a tensor of shape (batch, length, self.d_model) to
        (batch, length, self.d_output). The method can also accept an optional
        state parameter, which can be used for recurrent models.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, d_model).
            state (optional): Optional state information for recurrent models.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Output tensor of shape (batch, length, d_output).
                - state (optional): Updated state information, if applicable.

        Examples:
            >>> model = SequenceIdentity(d_model=128)
            >>> input_tensor = torch.randn(32, 10, 128)  # (batch, length, d_model)
            >>> output, state = model.forward(input_tensor)
            >>> output.shape
            torch.Size([32, 10, 128])

        Note:
            The state returned can vary depending on the specific implementation
            of the model, and may be None if no state is maintained.

        Raises:
            NotImplementedError: If the method is not implemented in a derived class.
        """
        return x, state

    def default_state(self, *batch_shape, device=None):
        """
        Create initial state for a batch of inputs.

        This method is designed to initialize the state of the sequence model
        based on the provided batch shape. The initial state can be used
        in recurrent operations within the model.

        Args:
            *batch_shape: Variable length argument list specifying the shape
                of the batch for which the state is to be initialized.
                Typically, this will include the batch size and any other
                dimensions relevant to the model.
            device (torch.device, optional): The device on which the state
                tensor should be created (e.g., 'cpu' or 'cuda'). If None,
                the default device will be used.

        Returns:
            torch.Tensor: A tensor representing the initial state, shaped
            according to the model's requirements. The content of the tensor
            will be determined by the specific implementation.

        Examples:
            >>> model = SequenceIdentity(d_model=128)
            >>> initial_state = model.default_state(32, device='cuda')
            >>> initial_state.shape
            torch.Size([32, 128])  # Example shape, actual shape may vary

        Note:
            The specific shape and content of the returned state tensor may
            vary depending on the model's architecture and requirements.
            It is expected that subclasses will override this method
            to provide an appropriate initial state.

        Raises:
            NotImplementedError: If the method is not implemented in
            subclasses of SequenceModule.
        """
        return None

    def step(self, x, state=None, **kwargs):
        """
            Abstract sequence model class.

        All models must adhere to this interface.

        A SequenceModule is generally a model that transforms an input of shape
        (n_batch, l_sequence, d_model) to (n_batch, l_sequence, d_output).

        REQUIRED methods and attributes:
        - forward, d_model, d_output: controls standard forward pass,
          a sequence-to-sequence transformation.
        - __init__ should also satisfy the following interface;
          see SequenceIdentity for an example:
            def __init__(self, d_model, transposed=False, **kwargs)

        OPTIONAL methods:
        - default_state, step: allows stepping the model recurrently with a hidden state.
        - state_to_tensor, d_state: allows decoding from hidden state.
        """
        return x, state
