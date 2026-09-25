"""Tests for espnet3/systems/esp2_s2t/whisper_token_id_converter.py.

The module is a copy of the espnet2 converter; these cover what the copy adds:
a ``token2id`` mapping that resolves the prompt and timestamp symbols, and
``keep_special_tokens``, which is what preserves the timestamps the serialized
output is scored on.
"""

import pytest

from espnet3.systems.esp2_s2t.whisper_token_id_converter import (
    OpenAIWhisperTokenIDConverter,
)

pytest.importorskip("whisper")


@pytest.fixture(params=["whisper_multilingual"])
def whisper_token_id_converter(request):
    return OpenAIWhisperTokenIDConverter(request.param)


def test_the_vocabulary_size_does_not_depend_on_who_ran_first():
    """50257 base + 107 specials + 1501 timestamps.

    This used to read 51867, which only held while an earlier test had grown
    the shared tokenizer through the added_tokens_txt path; that path now
    copies first.
    """
    converter = OpenAIWhisperTokenIDConverter("whisper_multilingual")
    assert converter.get_num_vocabulary_size() == 51865


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
