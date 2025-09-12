from importlib.metadata import version as pkg_version
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from packaging.version import Version
from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer

try:
    from transformers import AutoProcessor

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class Qwen2AudioTokenizer(AbsTokenizer):
    """Qwen2-Audio tokenizer that handles both text and audio inputs"""

    @typechecked
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
    ):
        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.model_name = model_name

        # Initialize processor lazily to avoid pickling issues
        self.processor = None

    def _build_processor(self):
        """Build AutoProcessor lazily to avoid serialization issues"""
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_name)

    def text2tokens(self, line: str) -> List[str]:
        """Convert text to tokens using Qwen2-Audio processor"""
        self._build_processor()

        # For text-only input, use standard tokenization
        tokens = self.processor.tokenizer.tokenize(line)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """Convert tokens back to text"""
        self._build_processor()
        return self.processor.tokenizer.convert_tokens_to_string(list(tokens))

    def create_multimodal_query(
        self,
        text_input: str,
        audio_input: Optional[Tuple[List[np.ndarray], int]] = None,
    ) -> Dict:
        """Create query with both text and audio inputs for Qwen2-Audio.

        This is the core tokenization process from the original example.
        """
        self._build_processor()

        # Handle audio input if provided
        if audio_input is not None:
            audios, sr = audio_input
            # The audio URLs are just placeholders for the chat template.
            # The actual audio data is passed in the `audios` argument below.
            wavs_query = [
                {"type": "audio", "audio_url": f"placeholder_{i}.wav"}
                for i in range(len(audios))
            ]

            text_query = [{"type": "text", "text": text_input}]

            query = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": wavs_query + text_query},
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                query, add_generation_prompt=True, tokenize=False
            )

            # Process inputs with both text and audio
            tfm_ver = Version(pkg_version("transformers"))
            if tfm_ver >= Version("4.55.0"):
                inputs = self.processor(
                    text=text,
                    audio=audios,
                    sampling_rate=sr,
                    return_tensors="np",
                    padding=True,
                )
            else:
                inputs = self.processor(
                    text=text,
                    audios=audios,
                    sampling_rate=sr,
                    return_tensors="np",
                    padding=True,
                )

        else:
            # Text-only processing
            inputs = self.processor(text=text_input, return_tensors="np", padding=True)

        return inputs
