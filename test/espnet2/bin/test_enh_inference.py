from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch
import yaml

from espnet2.bin.enh_inference import get_parser
from espnet2.bin.enh_inference import main
from espnet2.bin.enh_inference import SeparateSpeech
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.tasks.enh import EnhancementTask
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
            str(tmp_path),
        ]
    )

    with open(tmp_path / "config.yaml", "r") as f:
        args = yaml.safe_load(f)

    if args["encoder"] == "stft" and len(args["encoder_conf"]) == 0:
        args["encoder_conf"] = get_default_kwargs(STFTEncoder)

    with open(tmp_path / "config.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)

    return tmp_path / "config.yaml"


@pytest.fixture()
def inference_config(tmp_path: Path):
    # Write default configuration file
    args = {
        "encoder": "stft",
        "encoder_conf": {"n_fft": 64, "hop_length": 32},
        "decoder": "stft",
        "decoder_conf": {"n_fft": 64, "hop_length": 32},
    }
    with open(tmp_path / "inference.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)
    return tmp_path / "inference.yaml"


@pytest.fixture()
def invalid_inference_config(tmp_path: Path):
    # Write default configuration file
    args = {
        "encoder": "stft",
        "encoder_conf": {"n_fft": 64, "hop_length": 32},
        "xxx": "invalid",
    }
    with open(tmp_path / "invalid_inference.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)
    return tmp_path / "invalid_inference.yaml"


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


@pytest.mark.execution_timeout(5)
def test_SeparateSpeech_with_inference_config(config_file, inference_config):
    separate_speech = SeparateSpeech(
        train_config=config_file, inference_config=inference_config
    )
    wav = torch.rand(2, 16000)
    separate_speech(wav, fs=8000)


def test_SeparateSpeech_invalid_inference_config(
    inference_config, invalid_inference_config
):
    with pytest.raises(AssertionError):
        separate_speech = SeparateSpeech(
            train_config=None, model_file=None, inference_config=inference_config
        )

    with pytest.raises(AssertionError):
        separate_speech = SeparateSpeech(
            train_config=None, inference_config=invalid_inference_config
        )
