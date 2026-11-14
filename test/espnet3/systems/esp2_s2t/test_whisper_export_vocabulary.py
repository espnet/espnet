"""Tests for espnet3/systems/esp2_s2t/src/whisper_export_vocabulary.py.

The module is a copy of the espnet2 exporter; these cover what the copy
changes: a speaker-change symbol the vocabulary already holds is not appended
twice, and an added-token file does not follow the shared, cached tokenizer
into the next caller.
"""

import pytest

from espnet3.systems.esp2_s2t.src.whisper_export_vocabulary import export_vocabulary

pytest.importorskip("whisper")


@pytest.mark.execution_timeout(30)
def test_sot_asr_does_not_append_a_symbol_the_vocabulary_already_has(tmp_path):
    """A symbol already in the vocabulary is not appended again.

    "????" is Whisper BPE id 25629; appending it a second time would
    duplicate the row.
    """
    out = tmp_path / "tokens.txt"
    export_vocabulary(
        output=str(out),
        whisper_model="whisper_multilingual",
        log_level="INFO",
        sot_asr=True,
        speaker_change_symbol="????",
    )
    tokens = out.read_text().splitlines()
    assert tokens.count("????") == 1
    assert tokens.index("????") == 25629
    assert len(tokens) == 51865


@pytest.mark.execution_timeout(30)
def test_sot_asr_still_appends_a_symbol_the_vocabulary_lacks(tmp_path):
    out = tmp_path / "tokens.txt"
    export_vocabulary(
        output=str(out),
        whisper_model="whisper_multilingual",
        log_level="INFO",
        sot_asr=True,
        speaker_change_symbol="<sc>",
    )
    tokens = out.read_text().splitlines()
    assert tokens[-1] == "<sc>"
    assert len(tokens) == 51866
    assert tokens[51864] == "<|30.00|>"


@pytest.mark.execution_timeout(30)
def test_an_added_token_file_does_not_follow_the_next_caller(tmp_path):
    """add_token_file_name must not reach the shared tokenizer.

    whisper.tokenizer.get_tokenizer is lru_cached, so one HuggingFace
    tokenizer is shared by everything in the process. Adding to it in place
    made the exported length depend on who ran first: the padding loop reads
    the tokenizer's size, so every leaked token cost one timestamp at the
    end of the list.
    """
    add = tmp_path / "add.txt"
    add.write_text("command:yes\n", encoding="utf-8")
    polluter = tmp_path / "with_added.txt"
    export_vocabulary(
        output=str(polluter),
        whisper_model="whisper_multilingual",
        log_level="INFO",
        add_token_file_name=str(add),
    )

    after = tmp_path / "after.txt"
    export_vocabulary(
        output=str(after),
        whisper_model="whisper_multilingual",
        log_level="INFO",
    )
    tokens = after.read_text().splitlines()
    assert "command:yes" not in tokens
    assert len(tokens) == 51865
    assert tokens[51864] == "<|30.00|>"
