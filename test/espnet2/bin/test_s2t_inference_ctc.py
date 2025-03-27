from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.s2t_inference_ctc import (
    Speech2Text,
    Speech2TextGreedySearch,
    get_parser,
    main,
)
from espnet2.tasks.s2t_ctc import S2TTask
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
            "<nolang>",
            "<eng>",
            "<zho>",
            "<asr>",
            "<st_eng>",
            "a",
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
            "--promptencoder_conf",
            "output_size=4",
            "--preprocessor_conf",
            "fs=2000",
            "--preprocessor_conf",
            "speech_length=3",
            "--frontend_conf",
            "fs=16k",
            "--frontend_conf",
            "hop_length=160",
            "--encoder_conf",
            "input_layer=conv2d8",
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
    speech = np.random.randn(3000)
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
    speech = np.random.randn(3000)
    results = speech2text(
        speech,
        text_prev="<na>",
        lang_sym="<zho>",
        task_sym="<st_eng>",
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
    speech = np.random.randn(3000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_Speech2TextGreedy(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
        maxlenratio=-5,
    )
    speech = np.random.randn(3000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, _ in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)


@pytest.mark.execution_timeout(5)
def test_Speech2TextGreedy_overwrite_args(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
        maxlenratio=-5,
    )
    speech = np.random.randn(3000)
    results = speech2text(
        speech,
        text_prev="<na>",
        lang_sym="<zho>",
        task_sym="<st_eng>",
    )
    for text, token, token_int, text_nospecial, _ in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)


@pytest.mark.execution_timeout(5)
def test_Speech2TextGreedy_quantized(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
        maxlenratio=-5,
        quantize_s2t_model=True,
    )
    speech = np.random.randn(3000)
    results = speech2text(speech)
    for text, token, token_int, text_nospecial, _ in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(text_nospecial, str)


@pytest.mark.execution_timeout(5)
def test_Speech2TextGreedy_longform(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
        maxlenratio=-5,
    )
    speech = np.random.randn(3000)
    result = speech2text.decode_long_batched_buffered(
        speech,
        context_len_in_secs=1,
    )
    assert isinstance(result, str)


@pytest.mark.execution_timeout(5)
def test_Speech2TextGreedy_batchdecode(s2t_config_file):
    speech2text = Speech2TextGreedySearch(
        s2t_train_config=s2t_config_file,
    )
    result = speech2text.batch_decode(
        [
            np.random.randn(1000),
            np.random.randn(7000),
        ],
        batch_size=2,
        context_len_in_secs=1,
    )
    assert isinstance(result[0], str) and isinstance(result[1], str)
