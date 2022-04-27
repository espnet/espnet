from argparse import ArgumentParser
from pathlib import Path
import string

import numpy as np
import pytest

from espnet.nets.beam_search import Hypothesis
from espnet2.bin.st_inference import get_parser
from espnet2.bin.st_inference import main
from espnet2.bin.st_inference import Speech2Text
from espnet2.tasks.st import STTask


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
def src_token_list(tmp_path: Path):
    with (tmp_path / "src_tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "src_tokens.txt"


@pytest.fixture()
def st_config_file(tmp_path: Path, token_list, src_token_list):
    # Write default configuration file
    STTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "st"),
            "--token_list",
            str(token_list),
            "--src_token_list",
            str(src_token_list),
            "--token_type",
            "char",
        ]
    )
    return tmp_path / "st" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_Speech2Text(st_config_file):
    speech2text = Speech2Text(st_train_config=st_config_file, beam_size=1)
    speech = np.random.randn(1000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)
