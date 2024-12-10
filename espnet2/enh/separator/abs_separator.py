from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch


class AbsSeparator(torch.nn.Module, ABC):
    """
    Abstract base class for audio source separation models.

This class serves as a blueprint for creating audio source separation models. 
It inherits from PyTorch's `torch.nn.Module` and defines the necessary 
abstract methods that must be implemented in any subclass. This includes 
the `forward` method, which processes the input audio tensor, and the 
`num_spk` property, which indicates the number of speakers the model 
can separate.

Attributes:
    num_spk (int): The number of speakers that the separator can handle.

Args:
    input (torch.Tensor): Input tensor containing the audio data to be 
        processed.
    ilens (torch.Tensor): A tensor containing the lengths of the input 
        sequences.
    additional (Optional[Dict]): Optional additional information that may 
        be needed for processing.

Returns:
    Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]: A tuple 
        containing the separated audio tensors, a tensor for the lengths 
        of the separated signals, and an OrderedDict for any additional 
        outputs.

Yields:
    None

Raises:
    NotImplementedError: If the `forward` method is called directly on 
        this abstract class.

Examples:
    class MySeparator(AbsSeparator):
        def forward(self, input, ilens, additional=None):
            # Implement the separation logic here
            pass

        @property
        def num_spk(self):
            return 2  # For example, separating into 2 speakers

Note:
    This class should not be instantiated directly. It is meant to be 
    subclassed by concrete implementations of audio source separation models.

Todo:
    Implement concrete subclasses for specific separation algorithms.
    """
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        """
        Processes input data through the separator model.

    This method is responsible for performing the forward pass of the
    separator model, taking in an input tensor along with its lengths
    and any additional parameters required for processing. The method
    returns the separated output tensors, the output lengths, and an
    ordered dictionary of additional information.

    Args:
        input (torch.Tensor): The input tensor containing the audio data
            to be processed. The shape should be (batch_size, sequence_length, 
            features).
        ilens (torch.Tensor): A tensor containing the lengths of the input 
            sequences. The shape should be (batch_size,).
        additional (Optional[Dict], optional): A dictionary containing any 
            additional parameters required for processing. Defaults to None.

    Returns:
        Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]: A tuple 
        containing:
            - A tuple of output tensors, each representing the separated 
              audio sources.
            - A tensor containing the lengths of the output sequences.
            - An ordered dictionary with additional information about the 
              processing.

    Raises:
        NotImplementedError: If the method is not implemented in a subclass.

    Examples:
        >>> separator = MySeparator()  # MySeparator should inherit from AbsSeparator
        >>> input_tensor = torch.randn(2, 100, 64)  # Batch of 2, 100 time steps, 64 features
        >>> ilens_tensor = torch.tensor([100, 80])  # Lengths of input sequences
        >>> output, output_lengths, additional_info = separator.forward(
        ...     input_tensor, ilens_tensor
        ... )

    Note:
        This method must be implemented in any subclass of AbsSeparator.
        """
        raise NotImplementedError

    def forward_streaming(
        self,
        input_frame: torch.Tensor,
        buffer=None,
    ):
        """
        Forward pass for streaming input through the separator model.

    This method processes a single frame of input data and may utilize an
    internal buffer to maintain state across streaming inputs. The specific
    implementation will determine how the input is handled and how the buffer
    is used.

    Args:
        input_frame (torch.Tensor): A tensor representing a single frame of input
            data to be processed by the model.
        buffer (Optional): An optional buffer that may store state information
            across multiple calls. The default is None, indicating no buffer
            is used.

    Returns:
        The output from the forward pass, which may include processed audio
        data or other relevant information depending on the implementation.

    Raises:
        NotImplementedError: If the method is not implemented in the subclass.

    Examples:
        >>> separator = MySeparator()  # Assuming MySeparator is a concrete class
        >>> input_frame = torch.randn(1, 256)  # Example input frame
        >>> output = separator.forward_streaming(input_frame)
        >>> print(output)

    Note:
        The behavior of this method is dependent on the specific implementation
        of the `AbsSeparator` subclass and how it handles streaming data.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        raise NotImplementedError
