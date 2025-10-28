# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Abstract class for multimodal input/output handling in SpeechLM.

This module defines the base interface for handling different modalities
(text, audio, vision) with support for both discrete (tokenized) and
continuous (feature-based) representations.
"""

from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.nn import Module


class AbsIO(ABC, Module):
    """Abstract base class for multimodal I/O processing.

    This class provides the interface for encoding and decoding different
    modalities, supporting both discrete (e.g., text tokens, discrete codes)
    and continuous (e.g., audio features, embeddings) representations.

    All methods are optional and can be implemented as needed based on the
    specific modality and use case. Methods not implemented will raise
    NotImplementedError.

    Key methods:
        Data Processing:
        - preprocess: CPU-based single item preprocessing for data loading
        - encode_batch: GPU-based batch encoding for model input
        - decode_batch: GPU-based batch decoding for model output

        Utilities:
        - find_length: CPU-based length statistics collection before training
        - copy_for_worker: Create lightweight copy for multiprocessing workers

        Modality Properties:
        - feature_dim: Feature dimension for continuous modalities
        - num_stream: Number of streams for discrete multi-stream modalities
        - get_vocabulary: Vocabulary for discrete tokenized modalities
        - get_stream_interval: Token ranges for multi-stream tokenizers
        - get_stream_weight: Loss weights for multi-stream training
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

    def preprocess(
        self, data: Any
    ) -> Tuple[np.ndarray, Optional[Tuple[int, np.ndarray]], np.ndarray]:
        """Preprocess single data item on CPU for multiprocessing data loading.

        This method is called during data loading in multiprocessing workers
        and performs all CPU-based preprocessing operations on individual data
        items before they are batched. This includes operations like tokenization,
        feature extraction, normalization, etc.

        Note: This runs on CPU only and processes single items (not batches).
        Batch processing is handled by encode_batch after data loading.

        Args:
            data: Single raw data item in original format

        Returns:
            Tuple of (seq, conti_feat, loss_mask):
                - seq: np.ndarray of shape [t_len, num_stream] to be placed in
                  training sequence. For continuous features, fill with zeros.
                - conti_feat: Optional tuple of (length, features) where features
                  is the continuous data with time dimension first. None if discrete.
                - loss_mask: Float np.ndarray specifying loss weight for each token
                  in seq, same shape as seq.
        """
        raise NotImplementedError

    def encode_batch(self, batch_data: List[Any]) -> Dict[str, Any]:
        """Encode pre-processed batch data for GPU-based batch processing.

        This method handles data that has already been processed into proper
        shape and is ready for efficient GPU batch computation.

        Args:
            batch_data: List of pre-processed data items already in proper shape

        Returns:
            Dictionary containing GPU-ready batched tensors:
                - 'data': Main encoded tensor [batch, seq_len, ...]
                - 'lengths': Sequence lengths [batch]
        """
        raise NotImplementedError

    def decode_batch(self, batch_encoded: Dict[str, Any]) -> List[Any]:
        """Decode GPU-batched tensors back to list of individual items.

        This method handles GPU-batched data from encode_batch and converts
        it back to a list of individual decoded items.

        Args:
            batch_encoded: Dictionary of GPU-batched tensors from encode_batch

        Returns:
            List of decoded data items in their individual format
        """
        raise NotImplementedError

    def find_length(self, data: Any) -> int:
        """Calculate sequence length for length statistics collection before training.

        This CPU-only method is used during the pre-training phase to collect
        length statistics of the dataset. It efficiently computes the expected
        sequence length without performing full encoding, allowing for proper
        batch organization and padding strategies.

        Note: This runs on CPU only and is called during length statistics
        collection phase, not during actual training.

        Args:
            data: Single raw input data in modality-specific format

        Returns:
            Expected sequence length after encoding
        """
        raise NotImplementedError

    def copy_for_worker(self) -> "AbsIO":
        """Create a lightweight copy for multiprocessing data loading workers.

        This method creates a deep copy of the object while excluding heavy
        components like torch models, reducing memory usage and ensuring the
        object can be safely distributed to multiprocessing data loading workers.

        The default implementation performs a shallow copy, which may not be
        sufficient for all use cases. Subclasses should override this method
        to properly handle their specific components, especially:
        - Excluding large torch.nn.Module components
        - Excluding CUDA tensors
        - Keeping only necessary CPU-based preprocessing components

        Returns:
            A lightweight copy suitable for multiprocessing workers
        """
        raise NotImplementedError

    def feature_dim(self) -> Optional[int]:
        """Get the feature dimension for continuous modalities.

        Returns:
            Feature dimension (e.g., 80 for mel-spectrogram, 768 for embeddings),
            None for discrete modalities
        """
        raise NotImplementedError

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

    def get_vocabulary(self) -> Optional[List[str]]:
        """Get the complete vocabulary list for discrete modalities.

        For multi-stream tokenizers, this returns the combined vocabulary
        across all streams.

        Returns:
            List of vocabulary tokens/symbols (e.g., ["<pad>", "<unk>", "the", ...]),
            None for continuous modalities
        """
        raise NotImplementedError

    def get_stream_interval(self) -> Optional[List[Tuple[int, int]]]:
        """Get the vocabulary index ranges for all streams.

        In multi-stream tokenizers, each stream uses a specific range of
        vocabulary indices. For example, stream 0 might use indices [0, 1023],
        stream 1 uses [1024, 2047], etc.

        Returns:
            List of tuples (start, end) for each stream's vocabulary range,
            None for continuous modalities
        """
        raise NotImplementedError

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
