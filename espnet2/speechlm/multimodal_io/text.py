"""HuggingFace tokenizer-based text I/O implementation."""

from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer

from espnet2.speechlm.multimodal_io.abs_io import AbsIO


class HuggingFaceTextIO(AbsIO):
    """Text I/O using HuggingFace tokenizers.

    This class implements text encoding/decoding using HuggingFace's
    pretrained tokenizers. Text is discrete with a single stream.
    """

    def __init__(self, tokenizer_name: str):
        """Initialize HuggingFace text tokenizer.

        Args:
            tokenizer_name: HuggingFace model name or path for the tokenizer
                           (e.g., "bert-base-uncased", "gpt2", "facebook/opt-125m")
        """
        super().__init__(modality="text", is_discrete=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_batch(self, batch_data: List[str]) -> Dict[str, np.ndarray]:
        """Encode a single text string into token IDs.

        Args:
            batch_data: List containing exactly one text string

        Returns:
            Dictionary containing:
                - 'data': Token IDs array [1, seq_len]
        """
        if len(batch_data) != 1:
            raise ValueError(f"Text encode_batch only accepts batch size 1, got {len(batch_data)}")

        text = batch_data[0]

        # Tokenize single text without padding
        encoded = self.tokenizer(
            text,
            truncation=True,
            return_tensors="np",
        )

        return {
            "data": encoded["input_ids"],  # Shape: [1, seq_len]
        }

    def decode_batch(self, batch_encoded: Dict[str, np.ndarray]) -> List[str]:
        """Decode a single sequence of token IDs back to text string.

        Args:
            batch_encoded: Dictionary containing 'data' with token IDs [1, seq_len]

        Returns:
            List containing one decoded text string
        """
        token_ids = batch_encoded["data"]

        if token_ids.shape[0] != 1:
            raise ValueError(f"Text decode_batch only accepts batch size 1, got {token_ids.shape[0]}")

        # Decode single sequence
        text = self.tokenizer.decode(
            token_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return [text]

    def find_length_batch(self, batch_data: List[str]) -> List[int]:
        """Calculate token sequence lengths without full encoding.

        Args:
            batch_data: List of text strings

        Returns:
            List of token counts after tokenization
        """
        if len(batch_data) != 1:
            raise ValueError(f"Text find_length_batch only accepts batch size 1, got {len(batch_data)}")

        text = batch_data[0]
        # Fast tokenization without creating tensors
        tokens = self.tokenizer.tokenize(text)
        return [len(tokens)]

    def feature_dim(self) -> Optional[int]:
        """Get feature dimension (None for discrete text modality).

        Returns:
            None (text is discrete, not continuous)
        """
        return None

    def num_stream(self) -> Optional[int]:
        """Get number of streams (1 for text).

        Returns:
            1 (text uses single stream)
        """
        return 1

    def get_vocabulary(self) -> Optional[List[str]]:
        """Get the complete vocabulary list of the tokenizer.

        Returns:
            List of all tokens in the vocabulary
        """
        # Get vocabulary from tokenizer
        vocab = self.tokenizer.get_vocab()
        # Sort by token ID to maintain consistent ordering
        sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
        # Return just the token strings in order
        return [token for token, _ in sorted_tokens]

    def get_stream_interval(self) -> Optional[List[tuple]]:
        """Get vocabulary intervals for all streams.

        Returns:
            List containing single tuple [(0, vocab_size)] for text's single stream
        """
        vocab_size = len(self.tokenizer.get_vocab())
        return [(0, vocab_size)]

    def get_stream_weight(self) -> Optional[List[float]]:
        """Get loss weights for all streams.

        Returns:
            List containing [1.0] for single text stream
        """
        return [1.0]