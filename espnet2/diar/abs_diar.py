from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsDiarization(torch.nn.Module, ABC):
    """
    Abstract base class for diarization models in the ESPnet framework.

    This class serves as a blueprint for implementing specific diarization
    models. It extends the PyTorch `torch.nn.Module` and requires subclasses
    to implement the `forward` and `forward_rawwav` methods.

    Attributes:
        None

    Args:
        input (torch.Tensor): The input tensor representing audio features.
        ilens (torch.Tensor): A tensor containing the lengths of the input
            sequences.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, OrderedDict]: A tuple containing:
            - Output tensor with diarization predictions.
            - Tensor with the lengths of the output sequences.
            - An `OrderedDict` containing additional information.

    Yields:
        None

    Raises:
        NotImplementedError: If the subclass does not implement the required
            abstract methods.

    Examples:
        Subclassing the AbsDiarization to implement a specific diarization
        model might look like this:

        ```python
        class MyDiarizationModel(AbsDiarization):
            def forward(self, input, ilens):
                # Implementation of the forward pass
                pass

            def forward_rawwav(self, input, ilens):
                # Implementation for raw waveform input
                pass
        ```

    Note:
        This class is intended to be subclassed, and cannot be instantiated
        directly.

    Todo:
        Implement specific diarization models that inherit from this class.
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
        Defines the forward method for the AbsDiarization class, which is an abstract
        base class for speaker diarization models.

        This method is intended to process input tensors and their respective lengths
        to produce output tensors along with additional information in an
        OrderedDict. The specific implementation of this method must be provided by
        subclasses that inherit from AbsDiarization.

        Args:
            input (torch.Tensor): A tensor containing the input data to be processed,
                typically representing audio features.
            ilens (torch.Tensor): A tensor containing the lengths of the input data
                sequences, which is used to inform the model about the valid portion
                of each sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, OrderedDict]: A tuple consisting of:
                - A tensor representing the output of the model.
                - A tensor representing the lengths of the output sequences.
                - An OrderedDict containing additional information relevant to the
                processing, such as speaker embeddings or intermediate features.

        Raises:
            NotImplementedError: This method must be implemented in a subclass, and
                calling it directly from this base class will raise this exception.

        Examples:
            Here is an example of how a subclass might implement the forward method:

            class MyDiarizationModel(AbsDiarization):
                def forward(self, input: torch.Tensor, ilens: torch.Tensor):
                    # Implement the forward pass logic here
                    output = ...
                    output_lengths = ...
                    additional_info = OrderedDict(...)
                    return output, output_lengths, additional_info
        """
        raise NotImplementedError

    @abstractmethod
    def forward_rawwav(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, OrderedDict]:
        """
        Performs the forward pass of the model using raw waveform input.

        This method is an abstract implementation that must be overridden in
        subclasses of `AbsDiarization`. It processes the raw audio waveform
        tensor along with its corresponding lengths tensor, and returns the
        model's outputs.

        Args:
            input (torch.Tensor): A tensor representing the raw audio waveform,
                typically of shape (batch_size, num_samples).
            ilens (torch.Tensor): A tensor containing the lengths of the input
                audio waveforms, typically of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, OrderedDict]: A tuple containing:
                - A tensor with the model's output features (shape:
                (batch_size, num_features)).
                - A tensor with the predicted lengths (shape:
                (batch_size,)).
                - An ordered dictionary containing additional information
                regarding the forward pass.

        Raises:
            NotImplementedError: If the method is not implemented in the
            subclass.

        Examples:
            >>> model = YourDiarizationModel()  # Replace with your model class
            >>> raw_audio = torch.randn(2, 16000)  # Example raw audio for 2 samples
            >>> lengths = torch.tensor([16000, 16000])  # Example lengths
            >>> output_features, predicted_lengths, extra_info = model.forward_rawwav(
            ...     raw_audio, lengths)

        Note:
            This method is expected to handle audio data directly,
            and subclasses should implement the logic for processing the
            raw audio input accordingly.
        """
        raise NotImplementedError
