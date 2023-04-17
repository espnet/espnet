import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.lm_inference import GenerateText, get_parser, main
from espnet2.tasks.lm import LMTask
from espnet.nets.beam_search import Hypothesis


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
def lm_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    LMTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "lm"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
        ]
    )
    return tmp_path / "lm" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_GenerateText(lm_config_file):
    generatetext = GenerateText(lm_train_config=lm_config_file, beam_size=1)
    text = np.random.randint(
        low=1, high=len(generatetext.lm_train_args.token_list) - 1, size=(10,)
    )
    results = generatetext(text)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_GenerateText_quantized(lm_config_file):
    generatetext = GenerateText(
        lm_train_config=lm_config_file,
        beam_size=1,
        quantize_lm=True,
    )
    text = np.random.randint(
        low=1, high=len(generatetext.lm_train_args.token_list) - 1, size=(10,)
    )
    results = generatetext(text)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)
