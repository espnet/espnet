from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsEnhancement(torch.nn.Module, ABC):
    """
    Abstract base class for audio enhancement models.

    This class serves as a blueprint for implementing audio enhancement
    algorithms. It inherits from `torch.nn.Module` and requires the
    implementation of the `forward` and `forward_rawwav` methods, which
    process input tensors representing audio signals and their lengths.

    Attributes:
        None

    Args:
        None

    Returns:
        None

    Yields:
        None

    Raises:
        NotImplementedError: If the derived class does not implement the
        required methods.

    Examples:
        To create a custom enhancement model, subclass `AbsEnhancement` and
        implement the required methods:

        ```python
        class MyEnhancementModel(AbsEnhancement):
            def forward(self, input: torch.Tensor, ilens: torch.Tensor) -> Tuple:
                # Implement the forward pass
                pass

            def forward_rawwav(self, input: torch.Tensor, ilens: torch.Tensor) -> Tuple:
                # Implement the raw waveform processing
                pass
        ```

    Note:
        The `output_size` method is commented out but can be implemented
        in derived classes if needed to provide the size of the output
        tensor.

    Todo:
        - Implement additional methods as needed for specific enhancement
        tasks.
    """
    # @abstractmethod
    # def output_size(self) -> int:
    #     raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, OrderedDict]:
        """
        Computes the forward pass of the enhancement model.

        This method takes an input tensor and its corresponding lengths and 
        produces an output tensor along with updated lengths and additional 
        information encapsulated in an OrderedDict. The specific 
        implementation of this method should be provided in a derived class 
        that inherits from `AbsEnhancement`.

        Args:
            input (torch.Tensor): The input tensor representing the audio signal 
                to be enhanced. It is expected to have a shape of (batch_size, 
                num_channels, signal_length).
            ilens (torch.Tensor): A tensor containing the lengths of the input 
                sequences. It should have a shape of (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, OrderedDict]: A tuple containing:
                - output (torch.Tensor): The enhanced audio signal tensor.
                - olens (torch.Tensor): A tensor with the lengths of the 
                output sequences.
                - info (OrderedDict): An OrderedDict containing any additional 
                information generated during the forward pass.

        Raises:
            NotImplementedError: If this method is called directly from the 
                abstract class without an overriding implementation in a subclass.

        Examples:
            >>> model = MyEnhancementModel()  # Assume MyEnhancementModel 
            >>> input_tensor = torch.randn(10, 1, 16000)  # Example input
            >>> ilens = torch.tensor([16000] * 10)  # All sequences have 
            >>> output, olens, info = model.forward(input_tensor, ilens)
        """
        raise NotImplementedError

    @abstractmethod
    def forward_rawwav(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, OrderedDict]:
        """
        Computes the forward pass for raw waveform input in the enhancement model.

        This method processes the input raw waveform tensor and its corresponding 
        lengths to produce the enhanced output, which can be further utilized for 
        tasks such as speech enhancement or signal processing.

        Args:
            input (torch.Tensor): A tensor containing the raw waveform input data. 
                                The shape should be (batch_size, sequence_length).
            ilens (torch.Tensor): A tensor containing the lengths of the input 
                                sequences. The shape should be (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, OrderedDict]: 
                A tuple containing:
                    - output (torch.Tensor): The enhanced output waveform tensor.
                    - output_lengths (torch.Tensor): A tensor containing the lengths 
                    of the output sequences.
                    - extras (OrderedDict): A dictionary containing any additional 
                    information or metrics computed during the forward pass.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            >>> model = MyEnhancementModel()
            >>> raw_waveform = torch.randn(2, 16000)  # Example raw waveform for 2 samples
            >>> lengths = torch.tensor([16000, 12000])  # Lengths of each sample
            >>> output, output_lengths, extras = model.forward_rawwav(raw_waveform, lengths)

        Note:
            This method must be implemented in any subclass of AbsEnhancement.
        """
        raise NotImplementedError
