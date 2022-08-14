import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.bin.slu_inference import Speech2Text, get_parser, main
from espnet2.tasks.lm import LMTask
from espnet2.tasks.slu import SLUTask
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
def asr_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    SLUTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr"),
            "--token_list",
            str(token_list),
            "--transcript_token_list",
            str(token_list),
            "--token_type",
            "char",
        ]
    )
    return tmp_path / "asr" / "config.yaml"


@pytest.mark.execution_timeout(50)
def test_Speech2Text(asr_config_file):
    speech2text = Speech2Text(asr_train_config=asr_config_file, beam_size=1)
    speech = np.random.randn(100000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(50)
def test_Speech2Text_transcript(asr_config_file):
    speech2text = Speech2Text(asr_train_config=asr_config_file)
    speech = np.random.randn(100000)
    transcript = torch.randint(2, 4, [1, 4], dtype=torch.long)
    results = speech2text(speech, transcript)

    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


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
def test_Speech2Text_lm(asr_config_file, lm_config_file):
    speech2text = Speech2Text(
        asr_train_config=asr_config_file, lm_train_config=lm_config_file, beam_size=1
    )
    speech = np.random.randn(100000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_Speech2Text_quantized(asr_config_file, lm_config_file):
    speech2text = Speech2Text(
        asr_train_config=asr_config_file,
        lm_train_config=lm_config_file,
        beam_size=1,
        quantize_asr_model=True,
        quantize_lm=True,
    )
    speech = np.random.randn(100000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)
