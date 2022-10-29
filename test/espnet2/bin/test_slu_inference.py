import string
from argparse import ArgumentParser
from distutils.version import LooseVersion
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.bin.slu_inference import Speech2Understand, get_parser, main
from espnet2.tasks.lm import LMTask
from espnet2.tasks.slu import SLUTask
from espnet.nets.beam_search import Hypothesis

is_torch_1_5_plus = LooseVersion(torch.__version__) >= LooseVersion("1.5.0")


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
def slu_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    SLUTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "slu"),
            "--token_list",
            str(token_list),
            "--transcript_token_list",
            str(token_list),
            "--token_type",
            "char",
        ]
    )
    return tmp_path / "slu" / "config.yaml"


@pytest.mark.execution_timeout(50)
def test_Speech2Understand(slu_config_file):
    speech2understand = Speech2Understand(slu_train_config=slu_config_file, beam_size=1)
    speech = np.random.randn(100000)
    results = speech2understand(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(50)
def test_Speech2Understand_transcript(slu_config_file):
    speech2understand = Speech2Understand(slu_train_config=slu_config_file)
    speech = np.random.randn(100000)
    transcript = torch.randint(2, 4, [1, 4], dtype=torch.long)
    results = speech2understand(speech, transcript)

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


@pytest.mark.execution_timeout(10)
@pytest.mark.parametrize(
    "use_lm, token_type",
    [
        (False, "char"),
        (True, "char"),
        (False, "bpe"),
        (False, None),
    ],
)
def test_Speech2Understand_lm(use_lm, token_type, slu_config_file, lm_config_file):
    speech2understand = Speech2Understand(
        slu_train_config=slu_config_file,
        lm_train_config=lm_config_file if use_lm else None,
        beam_size=1,
        token_type=token_type,
    )
    speech = np.random.randn(100000)
    results = speech2understand(speech)
    for text, token, token_int, hyp in results:
        assert text is None or isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_Speech2Understand_quantized(slu_config_file, lm_config_file):
    speech2understand = Speech2Understand(
        slu_train_config=slu_config_file,
        lm_train_config=lm_config_file,
        beam_size=1,
        quantize_asr_model=True,
        quantize_lm=True,
    )
    speech = np.random.randn(100000)
    results = speech2understand(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)
