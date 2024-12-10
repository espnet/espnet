# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text-to-speech abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class AbsTTS2(torch.nn.Module, ABC):
    """
    Abstract base class for Text-to-Speech (TTS) systems using discrete units.

    This class defines the interface for TTS systems, which must implement
    the `forward` and `inference` methods. The `forward` method calculates
    outputs and returns a loss tensor, while the `inference` method is used
    to return predicted outputs as a dictionary.

    Attributes:
        require_raw_speech (bool): Indicates if raw speech is required. Defaults to False.
        require_vocoder (bool): Indicates if vocoder is required. Defaults to True.

    Methods:
        forward(text, text_lengths, feats, feats_lengths, **kwargs):
            Calculate outputs and return the loss tensor.
        inference(text, **kwargs):
            Return predicted output as a dict.

    Raises:
        NotImplementedError: If the method is not implemented by a subclass.

    Examples:
        class MyTTS(AbsTTS2):
            def forward(self, text, text_lengths, feats, feats_lengths, **kwargs):
                # Implementation of forward method
                pass

            def inference(self, text, **kwargs):
                # Implementation of inference method
                pass

        my_tts = MyTTS()
        loss = my_tts.forward(text_tensor, text_lengths_tensor, feats_tensor, feats_lengths_tensor)
        output = my_tts.inference(text_tensor)
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

        This method takes the input text and its associated lengths, along with
        features and their lengths, and processes them through the TTS model.
        It outputs the generated audio features, a dictionary of additional
        outputs, and the loss tensor.

        Args:
            text (torch.Tensor): A tensor containing the input text sequences.
            text_lengths (torch.Tensor): A tensor containing the lengths of the
                input text sequences.
            feats (torch.Tensor): A tensor containing the input feature sequences.
            feats_lengths (torch.Tensor): A tensor containing the lengths of the
                input feature sequences.
            **kwargs: Additional keyword arguments for model-specific options.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]: A tuple
            containing:
                - A tensor of generated audio features.
                - A dictionary with additional outputs.
                - A tensor representing the loss.

        Raises:
            NotImplementedError: If the method is not implemented in a derived
                class.

        Examples:
            >>> model = SomeDerivedTTSModel()
            >>> text = torch.tensor([[1, 2, 3], [4, 5, 0]])
            >>> text_lengths = torch.tensor([3, 2])
            >>> feats = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
            >>> feats_lengths = torch.tensor([2, 2])
            >>> outputs, additional_outputs, loss = model.forward(
            ...     text, text_lengths, feats, feats_lengths
            ... )
        """
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        text: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict."""
        raise NotImplementedError

    @property
    def require_raw_speech(self):
        """Return whether or not raw_speech is required."""
        return False

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return True
