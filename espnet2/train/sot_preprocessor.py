"""SOT Whisper preprocessor using tiktoken.

Tokenizes SOT text using the OpenAI whisper package's tiktoken encoding
directly, avoiding the broken ESPnet OpenAIWhisperTokenizer and the
HuggingFace Transformers dependency.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from espnet2.train.preprocessor import AbsPreprocessor


class SOTWhisperPreprocessor(AbsPreprocessor):
    """Tiktoken-based preprocessor for SOT Whisper training.

    Parses SOT text containing timestamps (<|X.XX|>), speaker change
    tokens (<sc>), and regular text. Encodes using tiktoken directly
    from the whisper package.

    Token ID layout (multilingual Whisper):
        0-50256:      BPE text tokens
        50257:        <|endoftext|> (EOS)
        50258:        <|startoftranscript|> (SOS)
        50259-50358:  Language tokens (<|en|>=50259)
        50359:        <|transcribe|>
        50360:        <|translate|>
        50363:        <|notimestamps|>
        50364-51864:  Timestamps <|0.00|> to <|30.00|>
        51865+:       Added tokens (<sc>, speaker tokens, ...)
    """

    def __init__(
        self,
        train: bool,
        whisper_language: str = "en",
        whisper_task: str = "transcribe",
        added_tokens_txt: Optional[str] = None,
        added_tokens: Optional[List[str]] = None,
        whisper_model: str = "tiny",
    ):
        super().__init__(train)

        import whisper

        self.whisper_model = whisper_model
        self.whisper_language = whisper_language
        self.whisper_task = whisper_task

        # Get tiktoken encoding from whisper
        # Use num_languages=100 for v3/turbo (adds <|yue|>, base vocab 51866)
        num_languages = 100 if "v3" in whisper_model or "turbo" in whisper_model else 99
        tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True,
            num_languages=num_languages,
            language=whisper_language,
            task=whisper_task,
        )
        self.encoding = tokenizer.encoding

        # Special token IDs from the tokenizer
        self.eot_id = tokenizer.eot  # 50257
        self.sot_id = tokenizer.sot  # 50258
        self.timestamp_begin = tokenizer.timestamp_begin  # 50364

        # Build language + task prefix (no <|notimestamps|> for SOT)
        # sot_sequence = (sot, lang, task) — skip sot, keep lang + task
        self.prefix_ids = list(tokenizer.sot_sequence[1:])

        # Load added tokens from file or list
        self.added_tokens: List[str] = []
        if added_tokens_txt is not None:
            try:
                with open(added_tokens_txt) as f:
                    self.added_tokens = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                logging.warning(
                    f"added_tokens_txt not found: {added_tokens_txt}, "
                    f"falling back to ['<sc>']"
                )
                self.added_tokens = ["<sc>"]
        elif added_tokens is not None:
            self.added_tokens = list(added_tokens)
        else:
            self.added_tokens = ["<sc>"]

        # Map added tokens to IDs. If a token already exists in the
        # base tiktoken vocab as a single token, reuse its existing ID.
        # Otherwise assign a new ID after the base vocabulary.
        self.added_token_map: Dict[str, int] = {}
        base_id = self.encoding.n_vocab
        n_new = 0
        for token in self.added_tokens:
            try:
                existing_ids = self.encoding.encode(token, allowed_special="all")
                if len(existing_ids) == 1:
                    self.added_token_map[token] = existing_ids[0]
                    continue
            except Exception:
                pass
            self.added_token_map[token] = base_id + n_new
            n_new += 1

        self.vocab_size = base_id + n_new

        # Regex for parsing: timestamp tokens, added tokens, or text
        # Matches <|...|> patterns and added tokens like <sc>
        added_escaped = [re.escape(t) for t in self.added_tokens]
        added_pattern = "|".join(added_escaped) if added_escaped else None
        timestamp_pattern = r"<\|\d+\.\d+\|>"
        special_pattern = r"<\|[^|]+\|>"

        patterns = [timestamp_pattern, special_pattern]
        if added_pattern:
            patterns.append(added_pattern)
        self._split_re = re.compile("(" + "|".join(patterns) + ")")

        logging.info(
            f"SOTWhisperPreprocessor: prefix={self.prefix_ids}, "
            f"added_tokens={self.added_token_map}, "
            f"vocab_size={self.vocab_size}"
        )

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize SOT text into token IDs."""
        token_ids = []

        # Split text into segments: timestamps, added tokens, regular text
        segments = self._split_re.split(text)

        for segment in segments:
            if not segment:
                continue

            # Check if it's a timestamp token <|X.XX|>
            ts_match = re.fullmatch(r"<\|(\d+\.\d+)\|>", segment)
            if ts_match:
                time_val = float(ts_match.group(1))
                ts_id = self.timestamp_begin + round(time_val / 0.02)
                token_ids.append(ts_id)
                continue

            # Check if it's an added token
            if segment in self.added_token_map:
                token_ids.append(self.added_token_map[segment])
                continue

            # Check if it's <|endoftext|> — strip it (model adds EOS)
            if segment == "<|endoftext|>":
                continue

            # Check if it's another special token <|...|> — skip it
            # (language/task tokens are added via prefix)
            if re.fullmatch(r"<\|[^|]+\|>", segment):
                continue

            # Regular text — encode with tiktoken
            if segment.strip():
                encoded = self.encoding.encode(segment, allowed_special="all")
                token_ids.extend(encoded)

        return token_ids

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if "text" in data and isinstance(data["text"], str):
            text = data["text"]
            token_ids = self._tokenize(text)
            # Prepend prefix: [<|en|>, <|transcribe|>]
            token_ids = self.prefix_ids + token_ids
            data["text"] = np.array(token_ids, dtype=np.int64)
        return data

    @classmethod
    def generate_token_list(
        cls,
        output_path: str,
        added_tokens_txt: Optional[str] = None,
        added_tokens: Optional[List[str]] = None,
        num_languages: int = 99,
    ) -> None:
        """Generate ESPnet token list file from tiktoken encoding.

        Args:
            output_path: Path to write the token list file.
            added_tokens_txt: Path to file with added tokens (one per line).
            added_tokens: List of added token strings.
            num_languages: Number of language tokens to include.
                99 for whisper small/medium/large-v1/v2 (51865 base vocab).
                100 for large-v3/large-v3-turbo (51866 base vocab, adds <|yue|>).
        """
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True, num_languages=num_languages
        )
        encoding = tokenizer.encoding

        # Load added tokens
        extra_tokens: List[str] = []
        if added_tokens_txt is not None:
            with open(added_tokens_txt) as f:
                extra_tokens = [line.strip() for line in f if line.strip()]
        elif added_tokens is not None:
            extra_tokens = list(added_tokens)
        else:
            extra_tokens = ["<sc>"]

        # Base vocab size depends on num_languages
        base_vocab_size = encoding.n_vocab
        lines = []
        for i in range(base_vocab_size):
            try:
                token_bytes = encoding.decode_single_token_bytes(i)
                token_str = token_bytes.decode("utf-8", errors="replace")
            except Exception:
                token_str = f"<|byte_{i}|>"
            # Escape newlines and tabs
            token_str = token_str.replace("\n", "\\n").replace("\t", "\\t")
            if not token_str:
                token_str = f"<|empty_{i}|>"
            lines.append(token_str)

        # Append added tokens
        for token in extra_tokens:
            lines.append(token)

        with open(output_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        logging.info(
            f"Wrote {len(lines)} tokens to {output_path} "
            f"(base={base_vocab_size}, added={len(extra_tokens)})"
        )
