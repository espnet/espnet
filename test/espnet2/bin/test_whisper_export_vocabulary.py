from argparse import ArgumentParser

import pytest

from espnet2.bin.whisper_export_vocabulary import export_vocabulary, get_parser, main

pytest.importorskip("whisper")


VOCAB_SIZE_MULTILINGUAL = 51865
VOCAB_SIZE_EN = 51864


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_export_vocabulary_to_stdout():
    try:
        export_vocabulary("-", "whisper_en")
    except Exception as e:
        pytest.fail(f"exception thrown: {e}")


def test_export_multilinugal_vocabulary_to_stdout():
    try:
        export_vocabulary("-", "whisper_multilingual", "en", "transcribe", "INFO")
    except Exception as e:
        pytest.fail(f"exception thrown: {e}")


def test_export_multilingual_vocabulary_to_stdout():
    try:
        export_vocabulary("-", "whisper_multilingual", "en")
    except Exception as e:
        pytest.fail(f"exception thrown: {e}")


def test_export_vocabulary_en(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/whisper_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()

    export_vocabulary(str(tknlist_path), "whisper_en", "en")

    with open(tknlist_path) as f:
        lines = f.readlines()

    assert len(lines) == VOCAB_SIZE_EN


def test_export_vocabulary_multilingual(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/whisper_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()

    export_vocabulary(str(tknlist_path), "whisper_multilingual", "zh")

    with open(tknlist_path) as f:
        lines = f.readlines()

    assert len(lines) == VOCAB_SIZE_MULTILINGUAL


def test_export_vocabulary_translation(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/whisper_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()

    export_vocabulary(str(tknlist_path), "whisper_multilingual", "zh", "translate")

    with open(tknlist_path) as f:
        lines = f.readlines()

    assert len(lines) == VOCAB_SIZE_MULTILINGUAL


def test_export_vocabulary_model_invalid():
    with pytest.raises(ValueError):
        export_vocabulary("-", "whisper_abc")


def test_export_vocabulary_lang_invalid():
    with pytest.raises(ValueError):
        export_vocabulary("-", "whisper_multilingual", "abc")


def test_export_vocabulary_task_invalid():
    with pytest.raises(ValueError):
        export_vocabulary("-", "whisper_multilingual", "zh", "transcribe_abc")


def test_export_vocabulary_to_stdout_sot():
    try:
        export_vocabulary("-", "whisper_en", "en", sot_asr=True)
    except Exception as e:
        pytest.fail(f"exception thrown: {e}")


def test_main(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/whisper_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()

    main(
        cmd=[
            "--whisper_model",
            "whisper_multilingual",
            "--output",
            str(tknlist_path),
            "--whisper_language",
            "en",
            "--whisper_task",
            "transcribe",
        ]
    )

    with open(tknlist_path) as f:
        lines = f.readlines()

    assert len(lines) == VOCAB_SIZE_MULTILINGUAL


def test_main_add_token(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/whisper_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()
    add_tknlist_path = tmp_path / "tmp_token_list/add_token_list.txt"
    with open(add_tknlist_path, "w") as f:
        f.write("command:yes\n")

    main(
        cmd=[
            "--whisper_model",
            "whisper_multilingual",
            "--output",
            str(tknlist_path),
            "--whisper_language",
            "en",
            "--add_token_file_name",
            str(add_tknlist_path),
        ]
    )

    found = False
    with open(tknlist_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "command:yes":
                found = True

    assert found is True


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
