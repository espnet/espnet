from argparse import ArgumentParser
from distutils.version import LooseVersion
from pathlib import Path
import string

import numpy as np
import pytest
import torch

from espnet.nets.beam_search import Hypothesis
from espnet2.bin.enh_asr_inference import get_parser
from espnet2.bin.enh_asr_inference import main
from espnet2.bin.enh_asr_inference import Speech2Text
from espnet2.tasks.enh_asr import ASRTask
from espnet2.tasks.lm import LMTask

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")


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
def joint_model_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    ASRTask.main(
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
def test_Speech2Text(joint_model_config_file, lm_config_file):
    speech2text = Speech2Text(
        joint_train_config=joint_model_config_file,
        lm_train_config=lm_config_file,
        beam_size=1,
    )
    if not is_torch_1_2_plus:
        # torchaudio.functional.istft is only available with pytorch 1.2+
        return
    speech = np.random.randn(100000)
    speech_ref1 = np.random.randn(100000)
    speech_ref2 = np.random.randn(100000)
    results_list = speech2text(speech, speech_ref1=speech_ref1, speech_ref2=speech_ref2)
    for spk_idx, results in enumerate(results_list):
        for sdr, text, token, token_int, hyp in results:
            assert sdr is None or isinstance(sdr, float)
            assert isinstance(text, str)
            assert isinstance(token[0], str)
            assert isinstance(token_int[0], int)
            assert isinstance(hyp, Hypothesis)
