"""Tests for espnet3/systems/esp2_s2t/src/whisper_token_id_converter.py.

The module is a copy of the espnet2 converter; these cover what the copy adds:
a ``token2id`` mapping that resolves the prompt and timestamp symbols, and
``keep_special_tokens``, which is what preserves the timestamps the serialized
output is scored on.
"""

import pytest

from espnet3.systems.esp2_s2t.src.whisper_token_id_converter import (
    OpenAIWhisperTokenIDConverter,
)

pytest.importorskip("whisper")


@pytest.fixture(params=["whisper_multilingual"])
def whisper_token_id_converter(request):
    return OpenAIWhisperTokenIDConverter(request.param)


def test_an_added_token_file_does_not_follow_the_next_caller(tmp_path):
    """Adding tokens here must not grow the tokenizer everything else shares.

    ``whisper.tokenizer.get_tokenizer`` is lru_cached, so one HuggingFace
    tokenizer is shared by every caller in the process. Adding to it in place
    made the vocabulary size depend on run order: another caller's
    ``get_num_vocabulary_size()`` read 51866 or 51867 rather than 51865, and
    the exporter, which pads with timestamps up to a fixed total, dropped one
    timestamp per leaked token.

    The copy this module takes before it adds anything is what stops that. It
    does not make this module immune to a leak someone else caused: a copy of
    an already-grown tokenizer is still grown. It stops the leak at its source.
    """
    import whisper.tokenizer

    shared = whisper.tokenizer.get_tokenizer(
        multilingual=True, language="en", task="transcribe"
    )
    before = len(shared.tokenizer.get_added_vocab())

    add = tmp_path / "add.txt"
    add.write_text("a-token-this-caller-wants\n", encoding="utf-8")
    converter = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", added_tokens_txt=str(add)
    )

    assert "a-token-this-caller-wants" in converter.token2id
    assert len(shared.tokenizer.get_added_vocab()) == before
    assert (
        OpenAIWhisperTokenIDConverter("whisper_multilingual").get_num_vocabulary_size()
        == 51865
    )


def test_token2id(whisper_token_id_converter: OpenAIWhisperTokenIDConverter):
    token2id = whisper_token_id_converter.token2id
    assert token2id["<|startoftranscript|>"] == 50258
    assert token2id["<|en|>"] == 50259
    assert token2id["<|transcribe|>"] == 50359
    assert token2id["<|notimestamps|>"] == 50363
    first = token2id["<|0.00|>"]
    last = token2id["<|30.00|>"]
    assert token2id["<|0.02|>"] == first + 1
    assert last - first == 1500


@pytest.mark.parametrize("sym", ["????", "<sc>"])
def test_token2id_speaker_change_symbol(sym):
    converter = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", sot=True, speaker_change_symbol=sym
    )
    if sym == "????":
        # Already a single Whisper BPE token: sot must reuse that row, not
        # mint a new id past the vocabulary.
        assert converter.token2id[sym] == 25629
    else:
        # A symbol Whisper lacks lands above the last timestamp row (51864).
        assert converter.token2id[sym] > 51864


def test_ids2tokens_keep_special_tokens():
    dropped = OpenAIWhisperTokenIDConverter("whisper_multilingual")
    kept = OpenAIWhisperTokenIDConverter(
        "whisper_multilingual", keep_special_tokens=True
    )
    ids = kept.tokenizer.tokenizer.convert_tokens_to_ids(["<|0.00|>", "Hi", "<|0.02|>"])
    assert dropped.ids2tokens(ids) == ["Hi"]
    assert kept.ids2tokens(ids) == ["<|0.00|>", "Hi", "<|0.02|>"]
