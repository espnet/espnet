import string
from argparse import ArgumentParser
from distutils.version import LooseVersion
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from espnet2.asr_transducer.beam_search_transducer import Hypothesis
from espnet2.bin.asr_transducer_inference import Speech2Text, get_parser, main
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
    enc_body_conf = (
        "{'body_conf': [{'block_type': 'conformer',"
        " 'hidden_size': 4, 'linear_size': 4,"
        " 'conv_mod_kernel_size': 3}]}"
    )
    decoder_conf = "{'hidden_size': 4}"
    joint_net_conf = "{'joint_space_size': 4}"

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


@pytest.fixture(
    params=[
        "conv2d_branchformer",
        "vgg_branchformer",
        "conv2d_conformer",
        "vgg_conformer",
        "conv2d_ebranchformer",
        "vgg_ebranchformer",
    ]
)
def asr_stream_config_file(request, tmp_path: Path, token_list):
    main_type = request.param.split("_")[1]

    enc_body_conf = (
        "{'body_conf': [{'block_type': '%s', 'hidden_size': 4, "
        "'linear_size': 4, 'conv_mod_kernel_size': 3},"
        "{'block_type': 'conv1d', 'kernel_size': 2, 'output_size': 2, "
        "'batch_norm': True, 'relu': True}], "
        "'main_conf': {'dynamic_chunk_training': True',"
        "'short_chunk_size': 1, 'num_left_chunks': 1}}"
    ) % (main_type)

    if request.param.startswith("vgg"):
        enc_body_conf = enc_body_conf[:-1] + (",'input_conf': {'vgg_like': True}}")

    decoder_conf = "{'hidden_size': 4}"
    joint_net_conf = "{'joint_space_size': 4}"

    ASRTransducerTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr_stream"),
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
    return tmp_path / "asr_stream" / "config.yaml"


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
    speech = np.random.randn(10000)
    hyps = speech2text(speech)
    results = speech2text.hypotheses_to_results(hyps)

    for text, token, token_int, hyp in results:
        assert text is None or isinstance(text, str)
        assert isinstance(token, List)
        assert isinstance(token_int, List)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(10)
@pytest.mark.parametrize(
    "use_lm, token_type, beam_search_config, decoding_window, left_context",
    [
        (False, "char", {"search_type": "default"}, 160, 0),
        (True, "char", {"search_type": "default"}, 160, 1),
        (False, "bpe", {"search_type": "default"}, 160, 0),
        (False, None, {"search_type": "default"}, 160, 1),
        (False, "char", {"search_type": "default"}, 320, 0),
        (False, "char", {"search_type": "tsd"}, 160, 1),
        (False, "char", {"search_type": "maes"}, 160, 1),
    ],
)
def test_streaming_Speech2Text(
    use_lm,
    token_type,
    beam_search_config,
    decoding_window,
    left_context,
    asr_stream_config_file,
    lm_config_file,
):
    speech2text = Speech2Text(
        asr_train_config=asr_stream_config_file,
        lm_train_config=lm_config_file if use_lm else None,
        beam_size=2,
        beam_search_config=beam_search_config,
        token_type=token_type,
        streaming=True,
        decoding_window=decoding_window,
        left_context=left_context,
    )

    speech = np.random.randn(10000)

    decoding_samples = speech2text.audio_processor.decoding_samples
    decoding_steps = len(speech) // decoding_samples

    for i in range(0, decoding_steps + 1, 1):
        _start = i * decoding_samples

        if i == decoding_steps:
            hyps = speech2text.streaming_decode(
                speech[i * decoding_samples : len(speech)], is_final=True
            )
        else:
            speech2text.streaming_decode(
                speech[(i * decoding_samples) : _start + decoding_samples - 1],
                is_final=False,
            )

    results = speech2text.hypotheses_to_results(hyps)

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
