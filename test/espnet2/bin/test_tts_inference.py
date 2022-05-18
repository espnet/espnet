import string
from argparse import ArgumentParser
from pathlib import Path

import pytest

from espnet2.bin.tts_inference import Text2Speech, get_parser, main
from espnet2.tasks.tts import TTSTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def config_file(tmp_path: Path, token_list):
    # Write default configuration file
    TTSTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--cleaner",
            "none",
            "--g2p",
            "none",
            "--normalize",
            "none",
        ]
    )
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_Text2Speech(config_file):
    text2speech = Text2Speech(train_config=config_file)
    text = "aiueo"
    text2speech(text)
