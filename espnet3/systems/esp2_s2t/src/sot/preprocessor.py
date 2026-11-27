"""The preprocessor that serialized output training (SOT) recipes use.

``S2TPreprocessor`` was written for the SentencePiece vocabulary of OWSM, and
no other S2T recipe uses a Whisper vocabulary with it. This subclass changes
three things so that it works with Whisper. The changes are needed for any
Whisper model trained with ``S2TPreprocessor``, so they belong to this system
and not to a recipe.

First, ``OpenAIWhisperTokenIDConverter.tokens2ids`` always adds the decoder
prompt in front of the tokens, and ``S2TPreprocessor`` then drops the first
id. The SOT text already starts with its own language and task symbols, so
this would remove ``<|en|>`` and leave ``<|notimestamps|>`` in front of the
timestamps. The subclass adds one placeholder id instead, which is the id that
``S2TPreprocessor`` drops.

Second, the speaker-change symbol must stay one token after tokenization.
``????`` is id 25629 in the Whisper BPE vocabulary. BPE sees it after a space,
as ``Ġ????``, which is not in the vocabulary. The tokenizer therefore splits it
unless it is told to keep the symbol whole.

Third, ``S2TPreprocessor`` returns speech with the shape ``(samples, 1)``. A
Whisper encoder without a frontend passes that tensor to ``torch.stft``, which
does not accept the last axis. The subclass returns 1-D speech.

``CommonPreprocessor_multi`` cannot be used either. Its Whisper converter puts
``<|notimestamps|>`` in every prompt, so a target with timestamps would start
with ``<|en|><|transcribe|><|notimestamps|><|0.00|>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from espnet2.train.preprocessor import S2TPreprocessor
from espnet3.systems.esp2_s2t.src.sot.whisper_token_id_converter import (
    OpenAIWhisperTokenIDConverter,
)
from espnet3.systems.esp2_s2t.src.sot.whisper_tokenizer import OpenAIWhisperTokenizer


def build_tokens2ids(convert, placeholder: int):
    """Return a ``tokens2ids`` that adds ``placeholder`` as the first id.

    ``S2TPreprocessor`` drops the first converted id, where it expects a prompt
    id. The SOT text already starts with ``<language><task>``, so a placeholder
    goes into that slot, and no real symbol is lost.

    Args:
        convert: The tokenizer's own tokens-to-ids callable.
        placeholder: Id to occupy index 0.

    Returns:
        A callable with ``tokens2ids``'s signature.
    """

    def tokens2ids(tokens):
        return [placeholder] + convert(tokens)

    return tokens2ids


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


class SotS2TPreprocessor(S2TPreprocessor):
    """S2TPreprocessor for a Whisper vocabulary carrying a speaker-change token.

    Args:
        speaker_change_symbol: The separator. Required, because it belongs to
            the checkpoint: it must be one token after tokenization, or the
            model cannot emit it, and the wrong one trains quietly against a
            vocabulary row that means something else.
        whisper_language: Language symbol for the tokenizer, without the bars.
        whisper_task: Task symbol for the tokenizer, without the bars.
        whisper_model_type: Whisper vocabulary the tokenizer and the converter
            are built from.

    Raises:
        ValueError: When the separator does not tokenize to a single id.
    """

    def __init__(
        self,
        train: bool,
        token_type: str,
        token_list,
        bpemodel,
        speaker_change_symbol: str,
        *args,
        whisper_language: str = "en",
        whisper_task: str = "transcribe",
        whisper_model_type: str = "whisper_multilingual",
        **kwargs,
    ) -> None:
        """Build the preprocessor. See the class docstring for the arguments."""
        super().__init__(train, token_type, token_list, bpemodel, *args, **kwargs)

        shared = dict(
            model_type=whisper_model_type,
            language=whisper_language,
            task=whisper_task,
            sot=True,
            speaker_change_symbol=speaker_change_symbol,
        )
        self.tokenizer = OpenAIWhisperTokenizer(**shared)
        converter = OpenAIWhisperTokenIDConverter(**shared)

        # The base reads notime/first_time/last_time from token2id before this
        # runs, and those three resolve identically either way, so the swap is
        # safe. See build_tokens2ids for what it is for.
        placeholder = converter.token2id["<|startoftranscript|>"]
        convert = converter.tokenizer.tokenizer.convert_tokens_to_ids
        converter.tokens2ids = build_tokens2ids(convert, placeholder)
        self.token_id_converter = converter

        # The id that the model must output. The sot=True tokenizer registers
        # the symbol as an added special token, which makes any string one
        # token, so asking the tokenizer proves nothing. The id must point to
        # this symbol in the token list from which the model size is set.
        self.speaker_change_symbol = speaker_change_symbol
        self.speaker_change_id = convert(
            self.tokenizer.text2tokens(speaker_change_symbol)
        )[0]
        # The base class does not keep token_list, because
        # OpenAIWhisperTokenIDConverter builds its vocabulary from the
        # tokenizer. It is read here, because S2TTask sets the model size from
        # this list.
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
            data: Sample with ``speech`` and the S2T text fields. A
                ``utt_id`` field is removed, because the model does not take
                it. Inference and scoring use it.

        Returns:
            The processed sample: ``speech`` as a 1-D float32 waveform, and
            ``text``, ``text_prev`` and ``text_ctc`` as int64 token-id arrays.

        Examples:
            >>> sample = {  # doctest: +SKIP
            ...     "speech": np.zeros(480000, dtype="float32"),
            ...     "text": "<|en|><|transcribe|><|0.00|> hi<|0.50|>",
            ...     "text_prev": "<|nospeech|>",
            ...     "text_ctc": "<|nospeech|>",
            ... }
            >>> out = preprocessor("AMI-0.0-30.0", sample)  # doctest: +SKIP
            >>> {key: out[key].shape for key in sorted(out)}  # doctest: +SKIP
            {'speech': (480000,), 'text': (9,), 'text_ctc': (2,), 'text_prev': (2,)}
        """
        data = {key: value for key, value in data.items() if key != "utt_id"}
        data = super().__call__(uid, data)
        speech = data.get("speech")
        if speech is not None and speech.ndim == 2 and speech.shape[1] == 1:
            # The Whisper encoder computes its features with torch.stft, which
            # needs 1-D speech.
            data["speech"] = speech[:, 0]
        return data
