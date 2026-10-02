"""The preprocessor the AMI SOT recipe trains through.

``S2TPreprocessor`` is written for OWSM's SentencePiece vocabulary, and no
upstream S2T recipe uses a Whisper one, so three things need correcting here.
All three are properties of the Whisper combination rather than of
``S2TPreprocessor``, which is why they live in the recipe.

``OpenAIWhisperTokenIDConverter.tokens2ids`` prepends the decoder prompt
unconditionally, and the base class then drops the first id. S2T text carries
its own ``<language><task>``, so that surgery deletes ``<|en|>`` and leaves a
``<|notimestamps|>`` standing in front of real timestamps.

The speaker-change symbol has to survive tokenization. ``????`` is Whisper BPE
id 25629, but ``Ġ????`` -- the form BPE sees, because the separator always
follows a space -- is not in the vocabulary, so it is split unless the
tokenizer is told to keep it whole.

The base class pads speech to ``(samples, 1)``. A full-Whisper encoder has no
frontend, so that tensor reaches ``torch.stft``, which rejects the trailing
axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.text.whisper_tokenizer import OpenAIWhisperTokenizer
from espnet2.train.preprocessor import S2TPreprocessor


def _read_token_list(token_list):
    """Return the token list as a list of strings, from a path or a sequence.

    Args:
        token_list: Path to a one-token-per-line file, or a sequence.

    Returns:
        The tokens, in id order.
    """
    if isinstance(token_list, (str, Path)):
        with open(token_list, encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle]
    return list(token_list)


class AmiSotS2TPreprocessor(S2TPreprocessor):
    """S2TPreprocessor for a Whisper vocabulary carrying a speaker-change token.

    Args:
        speaker_change_symbol: The separator. It must be one token after
            tokenization, or the model cannot emit it. Defaults to the symbol
            the released checkpoint uses.
        whisper_language: Language symbol for the tokenizer, without the bars.
        whisper_task: Task symbol for the tokenizer, without the bars.

    Raises:
        ValueError: When the separator does not tokenize to a single id.
    """

    def __init__(
        self,
        train: bool,
        token_type: str,
        token_list,
        bpemodel,
        *args,
        speaker_change_symbol: str = "????",
        whisper_language: str = "en",
        whisper_task: str = "transcribe",
        **kwargs,
    ) -> None:
        super().__init__(train, token_type, token_list, bpemodel, *args, **kwargs)

        shared = dict(
            model_type="whisper_multilingual",
            language=whisper_language,
            task=whisper_task,
            sot=True,
            speaker_change_symbol=speaker_change_symbol,
        )
        self.tokenizer = OpenAIWhisperTokenizer(**shared)
        converter = OpenAIWhisperTokenIDConverter(**shared)

        # The base class drops index 0, expecting a prompt id there. Our text
        # supplies its own <language><task>, so stand a placeholder in that
        # slot rather than let a real symbol be eaten. The base reads
        # notime/first_time/last_time from token2id before this runs, and those
        # three resolve identically either way, so the swap is safe.
        placeholder = converter.token2id["<|startoftranscript|>"]
        convert = converter.tokenizer.tokenizer.convert_tokens_to_ids
        converter.tokens2ids = lambda tokens: [placeholder] + convert(tokens)
        self.token_id_converter = converter

        # The id the model must emit. Asking the sot=True tokenizer whether the
        # symbol is "one token" proves nothing: registering it as an added
        # special token makes ANY string one token. What matters is that this
        # id names this symbol in the token list the model is sized from.
        self.speaker_change_symbol = speaker_change_symbol
        self.speaker_change_id = convert(
            self.tokenizer.text2tokens(speaker_change_symbol)
        )[0]
        # The base class does not keep token_list: OpenAIWhisperTokenIDConverter
        # builds its vocabulary from the tokenizer and has no such attribute.
        # Read it here, because it is the list S2TTask sizes the model from.
        listed = _read_token_list(token_list)
        if self.speaker_change_id >= len(listed) or (
            listed[self.speaker_change_id] != speaker_change_symbol
        ):
            held = (
                listed[self.speaker_change_id]
                if self.speaker_change_id < len(listed)
                else "<past the end>"
            )
            raise ValueError(
                f"speaker_change_symbol {speaker_change_symbol!r} tokenizes to "
                f"id {self.speaker_change_id}, but the token list holds "
                f"{held!r} there (length {len(listed)}). The corpus and the "
                "vocabulary were built with different separators."
            )

    def __call__(self, uid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess one sample, leaving speech one-dimensional.

        Args:
            uid: Utterance id.
            data: Sample with ``speech`` and the S2T text fields.

        Returns:
            The processed sample.
        """
        data = super().__call__(uid, data)
        speech = data.get("speech")
        if speech is not None and speech.ndim == 2 and speech.shape[1] == 1:
            # A full-Whisper encoder has no frontend; torch.stft takes (T,).
            data["speech"] = speech[:, 0]
        return data
