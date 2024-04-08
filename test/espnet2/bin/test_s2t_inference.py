from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.s2t_inference import Speech2Text, get_parser, main
from espnet2.tasks.s2t import S2TTask
from espnet.nets.beam_search import Hypothesis


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
            "<eng>",
            "<zho>",
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
            "rnn",
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
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        maxlenratio=-5,
    )
    speech = np.random.randn(1000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_Speech2Text_overwrite_args(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        maxlenratio=-5,
    )
    speech = np.random.randn(1000)
    results = speech2text(
        speech,
        text_prev="<na>",
        lang_sym="<zho>",
        task_sym="<st_eng>",
        predict_time=True,
    )
    for text, token, token_int, text_nospecial, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_Speech2Text_quantized(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        maxlenratio=-5,
        quantize_s2t_model=True,
    )
    speech = np.random.randn(1000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)
        assert isinstance(hyp, Hypothesis)
