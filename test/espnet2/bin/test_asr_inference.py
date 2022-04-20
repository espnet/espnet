from argparse import ArgumentParser
from pathlib import Path
import string

import numpy as np
import pytest

from espnet.nets.beam_search import Hypothesis
from espnet2.bin.asr_inference import get_parser
from espnet2.bin.asr_inference import main
from espnet2.bin.asr_inference import Speech2Text
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.lm import LMTask


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
    ASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
        ]
    )
    return tmp_path / "asr" / "config.yaml"


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
def test_Speech2Text(asr_config_file, lm_config_file):
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


@pytest.fixture()
def asr_config_file_streaming(tmp_path: Path, token_list):
    # Write default configuration file
    ASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr_streaming"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--decoder",
            "transformer",
        ]
    )
    return tmp_path / "asr_streaming" / "config.yaml"


@pytest.mark.execution_timeout(10)
def test_Speech2Text_streaming(asr_config_file_streaming, lm_config_file):
    speech2text = Speech2Text(
        asr_train_config=asr_config_file_streaming,
        lm_train_config=lm_config_file,
        beam_size=1,
        streaming=True,
    )
    speech = np.random.randn(100000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.fixture()
def enh_asr_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    EnhS2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "enh_asr"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
        ]
    )
    return tmp_path / "enh_asr" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_EnhS2T_Speech2Text(enh_asr_config_file, lm_config_file):
    speech2text = Speech2Text(
        asr_train_config=enh_asr_config_file,
        lm_train_config=lm_config_file,
        beam_size=1,
        enh_s2t_task=True,
    )
    speech = np.random.randn(100000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)
