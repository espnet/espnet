from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsDecoder(torch.nn.Module, ABC):
    """
    Abstract base class for audio decoders in the ESPnet2 framework.

This class serves as a blueprint for implementing specific decoder 
modules for audio processing tasks. It defines the essential methods 
that must be implemented by any concrete subclass, including the 
forward pass and streaming operations.

Attributes:
    None

Args:
    None

Methods:
    forward: Abstract method for the forward pass of the decoder.
    forward_streaming: Abstract method for processing input in a 
        streaming manner.
    streaming_merge: Merges frame-level processed audio chunks for 
        streaming output.

Raises:
    NotImplementedError: If the abstract methods are not implemented 
        in the derived class.

Examples:
    # Example subclass implementation
    class MyDecoder(AbsDecoder):
        def forward(self, input, ilens, fs=None):
            # Implementation here
            pass

        def forward_streaming(self, input_frame):
            # Implementation here
            pass

        def streaming_merge(self, chunks, ilens=None):
            # Implementation here
            pass

Note:
    This class should not be instantiated directly. Instead, derive 
    from it and implement the abstract methods to create a functional 
    decoder.

Todo:
    Implement additional methods or attributes as needed in 
    subclasses for specific audio processing tasks.
    """
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        fs: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the decoder.

    This method processes the input tensor through the decoder network,
    producing the output tensor along with the corresponding lengths. It is 
    an abstract method that must be implemented in any subclass of 
    AbsDecoder.

    Args:
        input (torch.Tensor): The input tensor representing the audio features.
        ilens (torch.Tensor): A tensor containing the lengths of the input 
            sequences. This is used to handle variable-length inputs.
        fs (int, optional): The sampling frequency of the input audio. If 
            not provided, it defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output (torch.Tensor): The output tensor after processing 
              through the decoder.
            - output_lengths (torch.Tensor): A tensor containing the lengths 
              of the output sequences.

    Raises:
        NotImplementedError: If this method is called directly on an 
            instance of AbsDecoder, as it is meant to be implemented by 
            subclasses.

    Examples:
        # Example of using the forward method in a subclass
        class MyDecoder(AbsDecoder):
            def forward(self, input, ilens, fs=None):
                # Implement the decoder logic here
                output = ...
                output_lengths = ...
                return output, output_lengths

        decoder = MyDecoder()
        output, output_lengths = decoder.forward(input_tensor, input_lengths)
        """
        raise NotImplementedError

    def forward_streaming(self, input_frame: torch.Tensor):
        """
        Perform the forward pass for streaming input frames.

    This method is designed to handle streaming input for the decoder. 
    It processes a single input frame at a time, allowing for real-time 
    decoding of audio data. The specific implementation of how the 
    input frame is processed will be defined in the subclass that 
    inherits from `AbsDecoder`.

    Args:
        input_frame (torch.Tensor): A tensor representing a single frame of 
            input audio data. The expected shape is (B, frame_size), where 
            B is the batch size and frame_size is the number of features 
            per frame.

    Returns:
        torch.Tensor: The output of the decoder for the given input frame. 
            The shape and nature of this output will depend on the 
            specific implementation in the subclass.

    Raises:
        NotImplementedError: If the method is not implemented in the 
            subclass.

    Examples:
        # Example of using the forward_streaming method in a subclass
        class MyDecoder(AbsDecoder):
            def forward_streaming(self, input_frame: torch.Tensor):
                # Custom processing logic for input_frame
                return processed_output

        decoder = MyDecoder()
        input_frame = torch.randn(1, 256)  # Example input
        output = decoder.forward_streaming(input_frame)
        """
        raise NotImplementedError

    def streaming_merge(self, chunks: torch.Tensor, ilens: torch.tensor = None):
        """
        Stream merge.

        This method merges the frame-level processed audio chunks in a 
        streaming simulation. It is important to note that, in real 
        applications, the processed audio should be sent to the output 
        channel frame by frame. This function serves as a guide to manage 
        your streaming output buffer.

        Args:
            chunks (torch.Tensor): A tensor of shape (B, frame_size) 
                containing the processed audio chunks.
            ilens (torch.Tensor, optional): A tensor of shape (B,) 
                representing the lengths of each chunk. Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape (B, T) representing the 
            merged audio output.

        Examples:
            >>> decoder = AbsDecoder()  # Instantiate a concrete subclass
            >>> chunks = torch.randn(2, 10)  # Example processed chunks
            >>> ilens = torch.tensor([10, 10])  # Example lengths
            >>> merged_audio = decoder.streaming_merge(chunks, ilens)
            >>> print(merged_audio.shape)  # Output shape should be (2, T)

        Note:
            Ensure that the input tensor `chunks` is properly shaped and 
            that `ilens` corresponds to the actual lengths of the chunks 
            provided.

        Raises:
            NotImplementedError: This method must be implemented in a 
            subclass of AbsDecoder.
        """

        raise NotImplementedError
