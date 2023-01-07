from argparse import ArgumentParser

import pytest

from espnet2.bin.whisper_export_vocabulary import (
    get_parser, 
    main,
    export_vocabulary
)

VOCAB_SIZE_MULTILINGUAL = 51865
VOCAB_SIZE_EN = 51864

def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)

def test_export_vocabulary_to_stdout():
    try:
        export_vocabulary(
            "-",
            "whisper_en",
            "INFO"
        )
    except Exception as e:
        pytest.fail(f"exception thrown: {e}")

def test_export_vocabulary_en(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/whisper_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()

    export_vocabulary(
        str(tknlist_path),
        "whisper_en",
        "INFO"
    )

    with open(tknlist_path) as f:
        lines = f.readlines()
    
    assert len(lines) == VOCAB_SIZE_EN

def test_export_vocabulary_invalid():
    with pytest.raises(ValueError):
       export_vocabulary(
            "-",
            "whisper_aaa",
            "INFO"
        ) 

def test_main(tmp_path):
    tknlist_path = tmp_path / "tmp_token_list/whisper_token_list.txt"
    tknlist_path.parent.mkdir()
    tknlist_path.touch()

    main(
        cmd=[
            "--whisper_model",
            "whisper_multilingual",
            "--output",
            str(tknlist_path)
        ]
    )

    with open(tknlist_path) as f:
        lines = f.readlines()
    
    assert len(lines) == VOCAB_SIZE_MULTILINGUAL