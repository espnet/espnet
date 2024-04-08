import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.lm_inference import GenerateText, get_parser, inference, main
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
        f.write("<generatetext>\n")
        f.write("<generatespeech>\n")
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
            "--model",
            "lm_multitask",
        ]
    )
    return tmp_path / "lm" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_GenerateText(lm_config_file):
    generatetext = GenerateText(lm_train_config=lm_config_file, beam_size=1)

    # Test with np.ndarray input
    text = np.random.randint(
        low=1, high=len(generatetext.lm_train_args.token_list) - 1, size=(10,)
    )
    results = generatetext(text)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)

    # Test with str input
    text = "this is a test"
    results = generatetext(text)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_GenerateText_unconditioned(lm_config_file):
    generatetext = GenerateText(lm_train_config=lm_config_file, beam_size=1)
    results = generatetext("<generatetext>")
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


@pytest.fixture()
def text_file(tmp_path: Path):
    with (tmp_path / "text").open("w") as f:
        f.write("test-1 how are you")
        f.write("test-2 i am good")
    return tmp_path / "text"


@pytest.mark.execution_timeout(5)
def test_inference(tmp_path: Path, lm_config_file, text_file):
    kwargs = dict(
        output_dir=str(tmp_path / "decode"),
        maxlen=5,
        minlen=1,
        batch_size=1,
        dtype="float32",
        beam_size=1,
        ngpu=0,
        seed=0,
        ngram_weight=0.0,
        penalty=0.0,
        nbest=1,
        num_workers=1,
        log_level="INFO",
        data_path_and_name_and_type=[(str(text_file), "text", "text")],
        key_file=None,
        lm_train_config=str(lm_config_file),
        lm_file=None,
        word_lm_train_config=None,
        word_lm_file=None,
        ngram_file=None,
        model_tag=None,
        token_type=None,
        bpemodel=None,
        allow_variable_data_keys=False,
        quantize_lm=False,
        quantize_modules=["Linear"],
        quantize_dtype="int8",
    )
    inference(**kwargs)
