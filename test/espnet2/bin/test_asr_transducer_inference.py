from argparse import ArgumentParser
from distutils.version import LooseVersion
from pathlib import Path
import string
from typing import List

import numpy as np
import pytest
import torch

from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.bin.asr_transducer_inference import get_parser
from espnet2.bin.asr_transducer_inference import main
from espnet2.bin.asr_transducer_inference import Speech2Text
from espnet2.tasks.asr_transducer import ASRTransducerTask
from espnet2.tasks.lm import LMTask

is_torch_1_5_plus = LooseVersion(torch.__version__) >= LooseVersion("1.5.0")


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def output_dir(tmp_path: Path):
    return tmp_path / "asr"


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def asr_config_file(tmp_path: Path, token_list):
    enc_body_conf = "{'body_conf': [{'block_type': 'rnn', 'dim_hidden': 4}]}"
    decoder_conf = "{'dim_hidden': 4}"
    joint_net_conf = "{'dim_joint_space': 4}"

    ASRTransducerTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--encoder_conf",
            enc_body_conf,
            "--decoder",
            "rnn",
            "--decoder_conf",
            decoder_conf,
            "--joint_network_conf",
            joint_net_conf,
        ]
    )
    return tmp_path / "asr" / "config.yaml"


@pytest.fixture()
def lm_config_file(tmp_path: Path, token_list):
    lm_conf = "{'nlayers': 1, 'unit': 8}"

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
            "--lm_conf",
            lm_conf,
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
def test_Speech2Text(use_lm, token_type, asr_config_file, lm_config_file):
    speech2text = Speech2Text(
        asr_train_config=asr_config_file,
        lm_train_config=lm_config_file if use_lm else None,
        beam_size=1,
        token_type=token_type,
    )
    speech = np.random.randn(100000)
    results = speech2text(speech)

    for text, token, token_int, hyp in results:
        assert text is None or isinstance(text, str)
        assert isinstance(token, List)
        assert isinstance(token_int, List)
        assert isinstance(hyp, Hypothesis)


# TO DO: upload mini_an4 pre-trained model to huggingface for additional tests.
def test_pretrained_speech2Text(asr_config_file):
    speech2text = Speech2Text.from_pretrained(
        model_tag=None,
        asr_train_config=asr_config_file,
        beam_size=1,
    )

    speech = np.random.randn(100000)
    _ = speech2text(speech)


@pytest.mark.parametrize(
    "quantize_params",
    [
        {},
        {"quantize_modules": ["LSTM", "Linear"]},
        {"quantize_dtype": "float16"},
    ],
)
def test_Speech2Text_quantization(asr_config_file, lm_config_file, quantize_params):
    if not is_torch_1_5_plus and quantize_params.get("quantize_dtype") == "float16":
        with pytest.raises(ValueError):
            speech2text = Speech2Text(
                asr_train_config=asr_config_file,
                lm_train_config=None,
                beam_size=1,
                token_type="char",
                quantize_asr_model=True,
                **quantize_params,
            )
    else:
        speech2text = Speech2Text(
            asr_train_config=asr_config_file,
            lm_train_config=None,
            beam_size=1,
            token_type="char",
            quantize_asr_model=True,
            **quantize_params,
        )

        speech = np.random.randn(100000)
        _ = speech2text(speech)


def test_Speech2Text_quantization_wrong_module(asr_config_file, lm_config_file):
    with pytest.raises(ValueError):
        _ = Speech2Text(
            asr_train_config=asr_config_file,
            lm_train_config=None,
            beam_size=1,
            token_type="char",
            quantize_asr_model=True,
            quantize_modules=["foo"],
        )
