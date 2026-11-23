"""Tests for espnet3/systems/esp2_s2t/src/sot/whisper_export_vocabulary.py.

The module is a copy of the espnet2 exporter. These tests cover two changes in
the copy. A speaker-change symbol that is already in the vocabulary is not
appended again. An added-token file does not change the shared, cached tokenizer
for the next caller.
"""

import pytest

from espnet3.systems.esp2_s2t.src.sot.whisper_export_vocabulary import export_vocabulary

pytest.importorskip("whisper")


@pytest.mark.execution_timeout(30)
def test_sot_asr_does_not_append_a_symbol_the_vocabulary_already_has(tmp_path):
    """A symbol already in the vocabulary is not appended again.

    "????" is Whisper BPE id 25629. Appending it a second time would duplicate
    the row.
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
    """add_token_file_name does not change the shared tokenizer.

    whisper.tokenizer.get_tokenizer is lru_cached, so all code in the process
    shares one HuggingFace tokenizer. When tokens were added to it in place, the
    exported length depended on which caller ran first. The padding loop reads the
    tokenizer's size, so each token added by an earlier caller removed one
    timestamp from the end of the list.
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
