import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import yaml

from espnet2.bin.asr_inference import Speech2Text, get_parser, main
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.enh_s2t import EnhS2TTask
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


@pytest.mark.execution_timeout(5)
def test_Speech2Text(asr_config_file, lm_config_file):
    speech2text = Speech2Text(
        asr_train_config=asr_config_file, lm_train_config=lm_config_file, beam_size=1
    )
    speech = np.random.randn(100000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(5)
def test_Speech2Text_quantized(asr_config_file, lm_config_file):
    speech2text = Speech2Text(
        asr_train_config=asr_config_file,
        lm_train_config=lm_config_file,
        beam_size=1,
        quantize_asr_model=True,
        quantize_lm=True,
    )
    speech = np.random.randn(100000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)


@pytest.fixture()
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
            "--encoder",
            "contextual_block_transformer",
        ]
    )
    return tmp_path / "asr_streaming" / "config.yaml"


@pytest.mark.execution_timeout(20)
def test_Speech2Text_streaming(asr_config_file_streaming, lm_config_file):
    file = open(asr_config_file_streaming, "r", encoding="utf-8")
    asr_train_config = file.read()
    asr_train_config = yaml.full_load(asr_train_config)
    asr_train_config["frontend"] = "default"
    asr_train_config["encoder_conf"] = {
        "look_ahead": 16,
        "hop_size": 16,
        "block_size": 40,
    }
    # Change the configuration file
    with open(asr_config_file_streaming, "w", encoding="utf-8") as files:
        yaml.dump(asr_train_config, files)
    speech2text = Speech2TextStreaming(
        asr_train_config=asr_config_file_streaming,
        lm_train_config=lm_config_file,
        beam_size=1,
    )
    speech = np.random.randn(10000)
    for sim_chunk_length in [1, 32, 128, 512, 1024, 2048]:
        if (len(speech) // sim_chunk_length) > 1:
            for i in range(len(speech) // sim_chunk_length):
                speech2text(
                    speech=speech[i * sim_chunk_length : (i + 1) * sim_chunk_length],
                    is_final=False,
                )
            results = speech2text(
                speech[(i + 1) * sim_chunk_length : len(speech)], is_final=True
            )
        else:
            results = speech2text(speech)
        for text, token, token_int, hyp in results:
            assert isinstance(text, str)
            assert isinstance(token[0], str)
            assert isinstance(token_int[0], int)
            assert isinstance(hyp, Hypothesis)

    # Test edge case: https://github.com/espnet/espnet/pull/4216
    file = open(asr_config_file_streaming, "r", encoding="utf-8")
    asr_train_config = file.read()
    asr_train_config = yaml.full_load(asr_train_config)
    asr_train_config["frontend"] = "default"
    asr_train_config["frontend_conf"] = {
        "n_fft": 256,
        "win_length": 256,
        "hop_length": 128,
    }
    # Change the configuration file
    with open(asr_config_file_streaming, "w", encoding="utf-8") as files:
        yaml.dump(asr_train_config, files)
    speech2text = Speech2TextStreaming(
        asr_train_config=asr_config_file_streaming,
        lm_train_config=lm_config_file,
        beam_size=1,
    )
    # edge case: speech is exactly multiple of sim_chunk_length, e.g., 10240 = 5 x 2048
    speech = np.random.randn(10240)
    for sim_chunk_length in [1, 32, 64, 128, 512, 1024, 2048]:
        if (len(speech) // sim_chunk_length) > 1:
            for i in range(len(speech) // sim_chunk_length):
                speech2text(
                    speech=speech[i * sim_chunk_length : (i + 1) * sim_chunk_length],
                    is_final=False,
                )
            results = speech2text(
                speech[(i + 1) * sim_chunk_length : len(speech)], is_final=True
            )
        else:
            results = speech2text(speech)
        for text, token, token_int, hyp in results:
            assert isinstance(text, str)
            assert isinstance(token[0], str)
            assert isinstance(token_int[0], int)
            assert isinstance(hyp, Hypothesis)


@pytest.fixture()
def enh_asr_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    EnhS2TTask.main(
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


@pytest.mark.execution_timeout(5)
def test_EnhS2T_Speech2Text(enh_asr_config_file, lm_config_file):
    speech2text = Speech2Text(
        asr_train_config=enh_asr_config_file,
        lm_train_config=lm_config_file,
        beam_size=1,
        enh_s2t_task=True,
    )
    speech = np.random.randn(48000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)
