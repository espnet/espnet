# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text-to-speech abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class AbsSynthesizer(torch.nn.Module, ABC):
    """
    Abstract base class for Text-to-Speech (TTS) synthesizers.

    This class defines the core interface for TTS synthesizers, including methods
    for forward processing and inference. Any subclass must implement these methods
    to provide specific functionality for synthesizing speech from input states.

    Attributes:
        require_raw_speech (bool): Indicates whether raw speech input is required.
        require_vocoder (bool): Indicates whether a vocoder is required for synthesis.

    Methods:
        forward(input_states, input_states_lengths, feats, feats_lengths, **kwargs):
            Calculate outputs and return the loss tensor.
        inference(input_states, **kwargs):
            Return predicted output as a dict.

    Raises:
        NotImplementedError: If a subclass does not implement the required methods.

    Examples:
        class MySynthesizer(AbsSynthesizer):
            def forward(self, input_states, input_states_lengths, feats,
                        feats_lengths, **kwargs):
                # Implementation here
                pass

            def inference(self, input_states, **kwargs):
                # Implementation here
                pass

        synthesizer = MySynthesizer()
        print(synthesizer.require_raw_speech)  # Output: False
        print(synthesizer.require_vocoder)      # Output: True
    """

    @abstractmethod
    def forward(
        self,
        input_states: torch.Tensor,
        input_states_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
                Calculate outputs and return the loss tensor.

        This method is responsible for processing the input states and
        features to produce the model outputs, which include the loss
        tensor and any additional information as a dictionary. The
        input tensors must adhere to specific dimensions that correspond
        to the expected data format.

        Args:
            input_states (torch.Tensor): A tensor containing the input states.
            input_states_lengths (torch.Tensor): A tensor indicating the lengths
                of the input states.
            feats (torch.Tensor): A tensor containing the features for synthesis.
            feats_lengths (torch.Tensor): A tensor indicating the lengths of the
                features.
            **kwargs: Additional keyword arguments for specific configurations.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple
                containing the loss tensor, a dictionary of outputs, and the
                additional tensor required for processing.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Examples:
            >>> model = MySynthesizer()  # MySynthesizer should inherit from AbsSynthesizer
            >>> input_states = torch.randn(1, 10, 256)  # Example input tensor
            >>> input_states_lengths = torch.tensor([10])
            >>> feats = torch.randn(1, 20, 80)  # Example feature tensor
            >>> feats_lengths = torch.tensor([20])
            >>> loss, outputs, additional = model.forward(
            ...     input_states, input_states_lengths, feats, feats_lengths)
        """
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        input_states: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
                AbsSynthesizer is an abstract class for text-to-speech (TTS) synthesis models.

        This class defines the essential methods and properties that any TTS synthesizer
        should implement, including the forward method for calculating outputs and
        losses, and the inference method for generating predicted outputs. It also
        includes properties to specify requirements for raw speech and vocoder.

        Attributes:
            require_raw_speech (bool): Indicates whether raw speech is required.
            require_vocoder (bool): Indicates whether a vocoder is required.

        Methods:
            forward(input_states, input_states_lengths, feats, feats_lengths, **kwargs):
                Calculate outputs and return the loss tensor.
            inference(input_states, **kwargs):
                Return predicted output as a dict.

        Examples:
            class MySynthesizer(AbsSynthesizer):
                def forward(self, input_states, input_states_lengths, feats,
                            feats_lengths, **kwargs):
                    # Implement forward logic
                    pass

                def inference(self, input_states, **kwargs):
                    # Implement inference logic
                    return {"output": torch.tensor([1.0])}

            synthesizer = MySynthesizer()
            output = synthesizer.inference(torch.tensor([[0.0]]))
            print(output)

        Note:
            This class cannot be instantiated directly and must be subclassed.

        Todo:
            Implement additional methods or properties as needed for specific use cases.
        """
        raise NotImplementedError

    @property
    def require_raw_speech(self):
        """Return whether or not raw_speech is required."""
        return False

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return True
