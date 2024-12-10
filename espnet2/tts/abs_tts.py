# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text-to-speech abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class AbsTTS(torch.nn.Module, ABC):
    """
    Abstract base class for Text-to-Speech (TTS) models.

    This class defines the essential methods and properties that any TTS
    implementation should have. It inherits from `torch.nn.Module` and
    provides an interface for forward processing and inference of TTS
    models.

    Attributes:
        require_raw_speech (bool): Indicates whether raw speech is required
            for the TTS model. Default is False.
        require_vocoder (bool): Indicates whether a vocoder is required for
            the TTS model. Default is True.

    Methods:
        forward(text: torch.Tensor, text_lengths: torch.Tensor, feats:
            torch.Tensor, feats_lengths: torch.Tensor, **kwargs) ->
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
            Abstract method that must be implemented to calculate outputs
            and return the loss tensor.

        inference(text: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
            Abstract method that must be implemented to return the predicted
            output as a dictionary.

    Raises:
        NotImplementedError: If the abstract methods are not implemented
            in a subclass.

    Examples:
        To create a concrete TTS model, subclass AbsTTS and implement the
        abstract methods. Here is an example:

        ```python
        class MyTTS(AbsTTS):
            def forward(self, text, text_lengths, feats, feats_lengths, **kwargs):
                # Implementation of the forward method
                pass

            def inference(self, text, **kwargs):
                # Implementation of the inference method
                pass
        ```
    """

    @abstractmethod
    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
                Calculate outputs and return the loss tensor.

        This method is responsible for processing the input text and its corresponding
        features to compute the model's outputs and the associated loss. It is an
        abstract method that must be implemented by any subclass of AbsTTS.

        Args:
            text (torch.Tensor): A tensor representing the input text.
            text_lengths (torch.Tensor): A tensor containing the lengths of the input
                text sequences.
            feats (torch.Tensor): A tensor representing the feature inputs for the
                model.
            feats_lengths (torch.Tensor): A tensor containing the lengths of the
                feature sequences.
            **kwargs: Additional keyword arguments for flexibility in implementation.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple
            containing:
                - A tensor representing the output of the model.
                - A dictionary with additional outputs, where keys are output names
                  and values are corresponding tensors.
                - A tensor representing the loss.

        Raises:
            NotImplementedError: If this method is called directly without
            implementation in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        text: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
                Text-to-speech abstract class.

        This class serves as an abstract base class for text-to-speech (TTS) models.
        It defines the core methods that must be implemented by any subclass, ensuring
        that the subclasses adhere to a common interface for both training and
        inference.

        Methods:
            forward: Calculate outputs and return the loss tensor.
            inference: Return predicted output as a dict.

        Attributes:
            require_raw_speech (bool): Indicates whether raw speech is required.
            require_vocoder (bool): Indicates whether a vocoder is required.
        """
        raise NotImplementedError

    @property
    def require_raw_speech(self):
        return False

    @property
    def require_vocoder(self):
        return True
