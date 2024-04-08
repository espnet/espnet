from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.s2t_inference_language import Speech2Language, get_parser, main
from espnet2.tasks.s2t import S2TTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        tokens = [
            "<blank>",
            "<unk>",
            "<na>",
            "<nospeech>",
            "<abk>",
            "<zul>",
            "<asr>",
            "<st_eng>",
            "<st_zho>",
            "<notimestamps>",
            "<0.00>",
            "<1.00>",
            "a",
            "i",
            "<sos>",
            "<eos>",
            "<sop>",
        ]
        for tok in tokens:
            f.write(f"{tok}\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def s2t_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    S2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "s2t"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--decoder",
            "transformer",
            "--decoder_conf",
            "linear_units=2",
            "--decoder_conf",
            "num_blocks=1",
            "--preprocessor_conf",
            "notime_symbol='<notimestamps>'",
            "--preprocessor_conf",
            "first_time_symbol='<0.00>'",
            "--preprocessor_conf",
            "last_time_symbol='<1.00>'",
            "--preprocessor_conf",
            "fs=2000",
            "--preprocessor_conf",
            "speech_length=1",
        ]
    )
    return tmp_path / "s2t" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_Speech2Text(s2t_config_file):
    speech2language = Speech2Language(
        s2t_train_config=s2t_config_file,
    )
    speech = np.random.randn(1000)
    results = speech2language(speech)
    for lang, prob in results:
        assert isinstance(lang, str)
        assert isinstance(prob, float)


@pytest.mark.execution_timeout(5)
def test_Speech2Text_quantized(s2t_config_file):
    speech2language = Speech2Language(
        s2t_train_config=s2t_config_file,
        quantize_s2t_model=True,
    )
    speech = np.random.randn(1000)
    results = speech2language(speech)
    for lang, prob in results:
        assert isinstance(lang, str)
        assert isinstance(prob, float)
