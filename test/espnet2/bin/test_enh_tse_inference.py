from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch
import yaml

from espnet2.bin.enh_tse_inference import SeparateSpeech, get_parser, main
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask
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
    TargetSpeakerExtractionTask.main(
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
    separate_speech(wav, fs=8000, enroll_ref1=torch.rand(batch_size, 16000))


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
    separate_speech(wav, fs=8000, enroll_ref1=torch.rand(2, 8000))


def test_SeparateSpeech_invalid_inference_config(
    enh_inference_config, invalid_enh_inference_config
):
    with pytest.raises(AssertionError):
        SeparateSpeech(
            train_config=None, model_file=None, inference_config=enh_inference_config
        )

    with pytest.raises(AssertionError):
        SeparateSpeech(train_config=None, inference_config=invalid_enh_inference_config)
