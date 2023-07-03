import string
from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch
import yaml

from espnet2.bin.enh_inference import SeparateSpeech, get_parser, main
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def config_file(tmp_path: Path):
    # Write default configuration file
    EnhancementTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "enh"),
        ]
    )

    with open(tmp_path / "enh" / "config.yaml", "r") as f:
        args = yaml.safe_load(f)

    if args["encoder"] == "stft" and len(args["encoder_conf"]) == 0:
        args["encoder_conf"] = get_default_kwargs(STFTEncoder)

    with open(tmp_path / "enh" / "config.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)

    return tmp_path / "enh" / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "input_size, segment_size, hop_size, normalize_segment_scale",
    [(16000, None, None, False), (35000, 2.4, 0.8, False), (35000, 2.4, 0.8, True)],
)
def test_SeparateSpeech(
    config_file, batch_size, input_size, segment_size, hop_size, normalize_segment_scale
):
    separate_speech = SeparateSpeech(
        train_config=config_file,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
    )
    wav = torch.rand(batch_size, input_size)
    separate_speech(wav, fs=8000)


@pytest.fixture()
def enh_inference_config(tmp_path: Path):
    # Write default configuration file
    args = {
        "encoder": "stft",
        "encoder_conf": {"n_fft": 64, "hop_length": 32},
        "decoder": "stft",
        "decoder_conf": {"n_fft": 64, "hop_length": 32},
    }
    (tmp_path / "enh").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "enh" / "inference.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)
    return tmp_path / "enh" / "inference.yaml"


@pytest.fixture()
def invalid_enh_inference_config(tmp_path: Path):
    # Write default configuration file
    args = {
        "encoder": "stft",
        "encoder_conf": {"n_fft": 64, "hop_length": 32},
        "xxx": "invalid",
    }
    (tmp_path / "enh").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "enh" / "invalid_inference.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)
    return tmp_path / "enh" / "invalid_inference.yaml"


@pytest.mark.execution_timeout(5)
def test_SeparateSpeech_with_inference_config(config_file, enh_inference_config):
    separate_speech = SeparateSpeech(
        train_config=config_file, inference_config=enh_inference_config
    )
    wav = torch.rand(2, 16000)
    separate_speech(wav, fs=8000)


def test_SeparateSpeech_invalid_inference_config(
    enh_inference_config, invalid_enh_inference_config
):
    with pytest.raises(AssertionError):
        SeparateSpeech(
            train_config=None, model_file=None, inference_config=enh_inference_config
        )

    with pytest.raises(AssertionError):
        SeparateSpeech(train_config=None, inference_config=invalid_enh_inference_config)


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
def enh_s2t_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    EnhS2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "enh_s2t"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--asr_decoder",
            "rnn",
        ]
    )

    with open(tmp_path / "enh_s2t" / "config.yaml", "r") as f:
        args = yaml.safe_load(f)

    if args["enh_encoder"] == "stft" and len(args["enh_encoder_conf"]) == 0:
        args["enh_encoder_conf"] = get_default_kwargs(STFTEncoder)

    with open(tmp_path / "enh_s2t" / "config.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)

    return tmp_path / "enh_s2t" / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "input_size, segment_size, hop_size, normalize_segment_scale",
    [(16000, None, None, False), (35000, 2.4, 0.8, False), (35000, 2.4, 0.8, True)],
)
def test_enh_s2t_SeparateSpeech(
    enh_s2t_config_file,
    batch_size,
    input_size,
    segment_size,
    hop_size,
    normalize_segment_scale,
):
    separate_speech = SeparateSpeech(
        train_config=enh_s2t_config_file,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        enh_s2t_task=True,
    )
    wav = torch.rand(batch_size, input_size)
    separate_speech(wav, fs=8000)


@pytest.fixture()
def enh_s2t_inference_config(tmp_path: Path):
    # Write default configuration file
    args = {
        "enh_encoder": "stft",
        "enh_encoder_conf": {"n_fft": 64, "hop_length": 32},
        "enh_decoder": "stft",
        "enh_decoder_conf": {"n_fft": 64, "hop_length": 32},
    }
    (tmp_path / "enh_s2t").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "enh_s2t" / "inference.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)
    return tmp_path / "enh_s2t" / "inference.yaml"


@pytest.fixture()
def invalid_enh_s2t_inference_config(tmp_path: Path):
    # Write default configuration file
    args = {
        "enh_encoder": "stft",
        "enh_encoder_conf": {"n_fft": 64, "hop_length": 32},
        "xxx": "invalid",
    }
    (tmp_path / "enh_s2t").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "enh_s2t" / "invalid_inference.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)
    return tmp_path / "enh_s2t" / "invalid_inference.yaml"


@pytest.mark.execution_timeout(5)
def test_enh_s2t_SeparateSpeech_with_inference_config(
    enh_s2t_config_file, enh_s2t_inference_config
):
    separate_speech = SeparateSpeech(
        train_config=enh_s2t_config_file,
        inference_config=enh_s2t_inference_config,
        enh_s2t_task=True,
    )
    wav = torch.rand(2, 16000)
    separate_speech(wav, fs=8000)


def test_enh_s2t_SeparateSpeech_invalid_inference_config(
    enh_s2t_inference_config, invalid_enh_s2t_inference_config
):
    with pytest.raises(AssertionError):
        SeparateSpeech(
            train_config=None,
            model_file=None,
            inference_config=enh_s2t_inference_config,
            enh_s2t_task=True,
        )

    with pytest.raises(AssertionError):
        SeparateSpeech(
            train_config=None,
            inference_config=invalid_enh_s2t_inference_config,
            enh_s2t_task=True,
        )
