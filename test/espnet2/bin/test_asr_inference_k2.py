import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask

pytest.importorskip("k2")


def test_get_parser():
    from espnet2.bin.asr_inference_k2 import get_parser

    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    from espnet2.bin.asr_inference_k2 import main

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
            "--decoder",
            "rnn",
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


@pytest.mark.execution_timeout(5)
def test_k2Speech2Text(asr_config_file, lm_config_file):
    from espnet2.bin.asr_inference_k2 import k2Speech2Text

    k2speech2text = k2Speech2Text(
        asr_train_config=asr_config_file, lm_train_config=lm_config_file, beam_size=1
    )
    batch_size = 5
    num_samples = 100000
    speech = np.random.randn(batch_size, num_samples).astype("f")
    speech_lengths = np.repeat(num_samples, batch_size).astype(np.int_)
    batch = {"speech": speech, "speech_lengths": speech_lengths}
    results = k2speech2text(batch)
    for text, token, token_int, score in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(score, float)
