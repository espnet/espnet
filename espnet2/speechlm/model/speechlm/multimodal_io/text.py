# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HuggingFace tokenizer-based text I/O implementation"""

from typing import List, Tuple

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from .abs_io import AbsIO


class HuggingFaceTextIO(AbsIO):
    """Text I/O using HuggingFace tokenizers.

    This class provides text tokenization using HuggingFace's pretrained
    tokenizers. Text is discrete with a single stream.
    """

    def __init__(self, tokenizer_name: str):
        """Initialize HuggingFace text tokenizer.

        Args:
            tokenizer_name: HuggingFace model name or path
                           (e.g., "bert-base-uncased", "gpt2")
        """
        super().__init__(modality="text", is_discrete=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name

        # Get the actual vocabulary size from model config
        self.vocab_size = AutoConfig.from_pretrained(tokenizer_name).vocab_size

    def preprocess(self, data: str) -> Tuple[np.ndarray, None, np.ndarray]:
        """Tokenize single text string for data loading.

        Args:
            data: Single text string

        Returns:
            Tuple of (tokens, conti_feat, loss_mask):
                - tokens: Token IDs as numpy array [seq_len, 1]
                - conti_feat: None (text is discrete)
                - loss_mask: Loss weights [seq_len, 1], all 1.0
        """
        # Use same tokenization as find_length for consistency
        token_ids = self.tokenizer.encode(
            data, truncation=True, add_special_tokens=True
        )

        tokens = np.array(token_ids, dtype=np.int32).reshape(-1, 1)
        conti_feat = None
        loss_mask = (tokens * 0 + 1).astype(np.float32)

        return tokens, conti_feat, loss_mask

    def decode(self, tokens: np.ndarray) -> str:
        """Decode a 1D tensor of token IDs to text string.

        Args:
            tokens: 1D numpy array of token IDs [seq_len]

        Returns:
            Decoded text string
        """
        # Ensure 1D array
        if tokens.ndim != 1:
            raise ValueError(f"Expected 1D tensor, got shape {tokens.shape}")

        # Convert numpy array to list and decode
        text = self.tokenizer.decode(
            tokens.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return text

    def find_length(self, data: str) -> int:
        """Get token count for length statistics.

        Args:
            data: Text string

        Returns:
            Number of tokens after tokenization
        """
        token_ids = self.tokenizer.encode(
            data, truncation=True, add_special_tokens=True
        )
        return len(token_ids)

    def copy_for_worker(self) -> "HuggingFaceTextIO":
        """Create copy for multiprocessing workers.

        Returns:
            New instance with same tokenizer
        """
        return self.__class__(self.tokenizer_name)

    def num_stream(self) -> int:
        """Text uses single stream."""
        return 1

    def get_vocabulary(self) -> List[str]:
        """Get tokenizer vocabulary.

        Returns:
            List of all tokens, padded to model vocab size
        """
        vocab = self.tokenizer.get_vocab()
        sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
        vocab_list = [token for token, _ in sorted_tokens]

        # Pad vocabulary to match model embedding size
        while len(vocab_list) < self.vocab_size:
            vocab_list.append(f"<unused_{len(vocab_list)}>")

        return vocab_list

    def get_stream_interval(self) -> List[Tuple[int, int]]:
        """Get vocabulary range for single stream.

        Returns:
            [(0, vocab_size)] for text's single stream
        """
        return [(0, self.vocab_size)]

    def get_stream_weight(self) -> List[float]:
        """Get loss weight for single stream.

        Returns:
            [1.0] for single text stream
        """
        return [1.0]
