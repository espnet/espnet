"""Abstract class for multimodal input/output handling in SpeechLM.

This module defines the base interface for handling different modalities
(text, audio, vision) with support for both discrete (tokenized) and
continuous (feature-based) representations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from torch.nn import Module
import numpy as np


class AbsIO(ABC, Module):
    """Abstract base class for multimodal I/O processing.

    This class provides the interface for encoding and decoding different
    modalities, supporting both discrete (e.g., text tokens, discrete codes)
    and continuous (e.g., audio features, embeddings) representations.
    """

    def __init__(self, modality: str, is_discrete: bool):
        """Initialize the multimodal I/O handler.

        Args:
            modality: Type of modality (e.g., "text", "audio", "vision")
            is_discrete: True for discrete representations (tokens),
                        False for continuous representations (features)
        """
        super().__init__()  # Initialize Module
        self.modality = modality
        self.is_discrete = is_discrete

    @abstractmethod
    def encode_batch(self, batch_data: List[Any]) -> Dict[str, np.ndarray]:
        """Encode a batch of raw input data into model-ready arrays.

        Args:
            batch_data: List of raw input data in modality-specific format

        Returns:
            Dictionary containing batched arrays with keys such as:
                - 'data': Main encoded array [batch, seq_len, ...]
                - 'lengths': Sequence lengths [batch]
                - 'mask': Attention mask [batch, seq_len]
        """
        raise NotImplementedError

    @abstractmethod
    def decode_batch(self, batch_encoded: Dict[str, np.ndarray]) -> List[Any]:
        """Decode a batch of encoded arrays back to original format.

        Args:
            batch_encoded: Dictionary of batched arrays from encode_batch

        Returns:
            List of decoded data in original modality format
        """
        raise NotImplementedError

    @abstractmethod
    def find_length_batch(self, batch_data: List[Any]) -> List[int]:
        """Calculate sequence lengths after encoding without full encoding.

        This method allows efficient length calculation for batching
        and padding operations without performing the actual encoding.

        Args:
            batch_data: List of raw input data in modality-specific format

        Returns:
            List of expected sequence lengths after encoding
        """
        raise NotImplementedError

    @abstractmethod
    def feature_dim(self) -> Optional[int]:
        """Get the feature dimension for continuous modalities.

        Returns:
            Feature dimension (e.g., 80 for mel-spectrogram, 768 for embeddings),
            None for discrete modalities
        """
        raise NotImplementedError

    @abstractmethod
    def num_stream(self) -> Optional[int]:
        """Get the number of parallel streams for discrete modalities.

        For multi-stream discrete representations, tokens are organized as [T, N]
        where T is the sequence length and N is the number of parallel streams.
        Each stream represents a different aspect or level of the signal
        (e.g., semantic vs acoustic codes in audio).

        Returns:
            Number of parallel streams (e.g., 8 for multi-stream audio codes),
            None for continuous modalities
        """
        raise NotImplementedError

    @abstractmethod
    def get_vocabulary(self) -> Optional[List[str]]:
        """Get the complete vocabulary list for discrete modalities.

        For multi-stream tokenizers, this returns the combined vocabulary
        across all streams.

        Returns:
            List of vocabulary tokens/symbols (e.g., ["<pad>", "<unk>", "the", ...]),
            None for continuous modalities
        """
        raise NotImplementedError

    @abstractmethod
    def get_stream_interval(self) -> Optional[List[tuple]]:
        """Get the vocabulary index ranges for all streams.

        In multi-stream tokenizers, each stream uses a specific range of
        vocabulary indices. For example, stream 0 might use indices [0, 1023],
        stream 1 uses [1024, 2047], etc.

        Returns:
            List of tuples (start, end) for each stream's vocabulary range,
            None for continuous modalities
        """
        raise NotImplementedError

    @abstractmethod
    def get_stream_weight(self) -> Optional[List[float]]:
        """Get the loss weights for all streams.

        Different streams may have different importance during training.
        For example, semantic streams might be weighted higher than
        acoustic detail streams.

        Returns:
            List of weight values for each stream (typically between 0.0 and 1.0),
            None for continuous modalities
        """
        raise NotImplementedError