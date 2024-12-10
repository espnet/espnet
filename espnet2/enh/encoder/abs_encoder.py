from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsEncoder(torch.nn.Module, ABC):
    """
    Abstract base class for encoders in the ESPnet2 framework.

This class serves as the foundation for all encoder implementations within the
ESPnet2 framework. It defines the essential methods and properties that any 
specific encoder must implement, including the forward pass and output 
dimensions. This is a subclass of `torch.nn.Module` and follows the ABC 
pattern.

Attributes:
    output_dim (int): The dimensionality of the output from the encoder.

Methods:
    forward(input: torch.Tensor, ilens: torch.Tensor, fs: int = None) -> 
        Tuple[torch.Tensor, torch.Tensor]:
        Performs the forward pass of the encoder.

    forward_streaming(input: torch.Tensor):
        Performs the forward pass for streaming input.

    streaming_frame(audio: torch.Tensor):
        Splits the continuous audio into frame-level audio chunks in a 
        streaming simulation.

Args:
    input (torch.Tensor): The input tensor to the encoder.
    ilens (torch.Tensor): Lengths of the input sequences.
    fs (int, optional): Sampling frequency of the audio input. Defaults to None.
    audio (torch.Tensor): Continuous audio input for frame extraction.

Returns:
    Tuple[torch.Tensor, torch.Tensor]: The output tensor from the encoder and 
    the corresponding lengths.

Yields:
    None

Raises:
    NotImplementedError: If the method is not implemented in the subclass.

Examples:
    # Example of a subclass implementation
    class MyEncoder(AbsEncoder):
        def forward(self, input, ilens, fs=None):
            # Implementation of the forward method
            pass

        @property
        def output_dim(self):
            return 256  # Example output dimension

    # Creating an instance of MyEncoder
    encoder = MyEncoder()
    output, output_lengths = encoder(input_tensor, input_lengths)

Note:
    The `streaming_frame` method is designed for use in streaming 
    applications and assumes that the entire audio input is provided for 
    processing.

Todo:
    Implement concrete subclasses that define the `forward` and `output_dim` 
    methods.
    """
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        fs: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the encoder.

    This method processes the input tensor and the input lengths to produce the 
    output tensor and the corresponding lengths. The specific implementation 
    depends on the derived encoder class.

    Args:
        input (torch.Tensor): The input tensor of shape (B, T) where B is the 
            batch size and T is the sequence length.
        ilens (torch.Tensor): A tensor containing the lengths of each input 
            sequence in the batch, of shape (B,).
        fs (int, optional): Sampling frequency. If provided, it may be used 
            in the processing of the input. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Output tensor of shape (B, D) where D is the output dimension.
            - A tensor containing the lengths of the output sequences, of shape 
            (B,).

    Raises:
        NotImplementedError: If this method is not overridden in a derived class.

    Examples:
        >>> encoder = MyEncoder()  # MyEncoder is a subclass of AbsEncoder
        >>> input_tensor = torch.randn(32, 100)  # Batch of 32, sequence length 100
        >>> ilens_tensor = torch.tensor([100] * 32)  # All sequences are of length 100
        >>> output, output_lengths = encoder(input_tensor, ilens_tensor, fs=16000)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Abstract base class for encoders in a neural network model.

    This class defines the interface for encoder modules that process input
    tensors. It requires implementation of the `forward` method and the 
    `output_dim` property, which should return the dimension of the output 
    from the encoder.

    Attributes:
        output_dim (int): The dimension of the output from the encoder.

    Args:
        input (torch.Tensor): Input tensor to be processed by the encoder.
        ilens (torch.Tensor): Lengths of the input sequences.
        fs (int, optional): Sampling frequency. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output tensor
        and any additional output (e.g., hidden states or attention weights).

    Raises:
        NotImplementedError: If the `forward` method or `output_dim` property 
        is not implemented in a subclass.

    Examples:
        class MyEncoder(AbsEncoder):
            @property
            def output_dim(self) -> int:
                return 256

            def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
                # Implementation of the forward pass
                pass
        """
        raise NotImplementedError

    def forward_streaming(self, input: torch.Tensor):
        """
        Forward streaming method for the AbsEncoder class.

    This method is intended to process input tensors in a streaming manner,
    allowing for the encoding of data that is received in chunks rather than
    all at once. This is particularly useful in applications where data is
    generated or received continuously, such as in audio processing.

    Args:
        input: A torch.Tensor containing the input data to be processed in a
            streaming fashion. The shape of the tensor should be compatible
            with the encoder's expected input dimensions.

    Returns:
        A torch.Tensor representing the encoded output of the input data.
    
    Raises:
        NotImplementedError: This method should be implemented in subclasses
            of AbsEncoder.

    Examples:
        >>> encoder = MyEncoder()  # Assuming MyEncoder is a subclass of AbsEncoder
        >>> input_data = torch.randn(1, 16000)  # Example input tensor
        >>> output = encoder.forward_streaming(input_data)
        >>> print(output.shape)  # Expected output shape depends on encoder design
        """
        raise NotImplementedError

    def streaming_frame(self, audio: torch.Tensor):
        """
        Stream frame.

        This method splits continuous audio into frame-level audio chunks 
        simulating a streaming scenario. It is important to note that this 
        function takes the entire long audio as input for the simulation. 
        You may refer to this function to manage your streaming input buffer 
        in a real streaming application.

        Args:
            audio (torch.Tensor): A tensor of shape (B, T), where B is the 
                batch size and T is the length of the audio sequence.

        Returns:
            List[torch.Tensor]: A list of tensors, each of shape 
                (B, frame_size), representing the chunked audio frames.

        Examples:
            >>> encoder = AbsEncoder()
            >>> audio_input = torch.randn(2, 16000)  # Example with 2 audio samples
            >>> frames = encoder.streaming_frame(audio_input)
            >>> print(len(frames))  # Number of frames created
            >>> print(frames[0].shape)  # Shape of the first frame

        Note:
            Ensure that the audio tensor is appropriately formatted and 
            contains continuous audio data before passing it to this method.

        Raises:
            NotImplementedError: If the method is not implemented in a 
            subclass.
        """
        raise NotImplementedError
