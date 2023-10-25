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
