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
            "<en>",
            "<asr>",
            "<st_en>" "<notimestamps>",
            "<0.00>",
            "<30.00>",
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
        ]
    )
    return tmp_path / "s2t" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_Speech2Text(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file, beam_size=1, time_sym="<0.00>"
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
def test_Speech2Text_quantized(s2t_config_file):
    speech2text = Speech2Text(
        s2t_train_config=s2t_config_file,
        beam_size=1,
        time_sym="<0.00>",
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
