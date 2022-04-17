from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch

from espnet2.bin.diar_enh_inference import DiarSepSpeech
from espnet2.bin.diar_enh_inference import get_parser
from espnet2.bin.diar_enh_inference import main
from espnet2.tasks.diar_enh import DiarEnhTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def config_file(tmp_path: Path):
    # Write default configuration file
    DiarEnhTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--num_spk",
            "2",
        ]
    )
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "input_size, segment_size, hop_size, normalize_segment_scale, num_spk, multiply_diar_result",
    [
        (16000, None, None, False, 2, False),
        (35000, 2.4, 1.2, False, 2, False),
        (34000, 2.4, 1.2, True, 2, True),
    ],
)
def test_DiarSepSpeech(
    config_file,
    batch_size,
    input_size,
    segment_size,
    hop_size,
    normalize_segment_scale,
    num_spk,
    multiply_diar_result,
):
    diarize_speech = DiarSepSpeech(
        train_config=config_file,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        num_spk=num_spk,
        multiply_diar_result=multiply_diar_result,
    )
    wav = torch.rand(batch_size, input_size)
    diarize_speech(wav, fs=8000)


@pytest.fixture()
def config_file2(tmp_path: Path):
    # Write default configuration file
    DiarEnhTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--attractor",
            "rnn",
            "--attractor_conf",
            "unit=256",
            "--num_spk",
            "2",
        ]
    )
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "input_size2, segment_size2, hop_size2, normalize_segment_scale2, num_spk2",
    [
        (16000, None, None, False, None),
        (35000, 2.4, 1.2, False, None),
        (34000, 2.4, 1.2, True, None),
        (16000, None, None, False, 2),
        (35000, 2.4, 1.2, False, 2),
        (34000, 2.4, 1.2, True, 2),
    ],
)
def test_DiarSepSpeech2(
    config_file2,
    batch_size,
    input_size2,
    segment_size2,
    hop_size2,
    normalize_segment_scale2,
    num_spk2,
):
    diarize_speech = DiarSepSpeech(
        train_config=config_file2,
        segment_size=segment_size2,
        hop_size=hop_size2,
        normalize_segment_scale=normalize_segment_scale2,
        num_spk=num_spk2,
    )
    wav = torch.rand(batch_size, input_size2)
    diarize_speech(wav, fs=8000)


@pytest.fixture()
def config_file3(tmp_path: Path):
    # Write default configuration file
    DiarEnhTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--num_spk",
            "2",
            "--frontend",
            "default",
        ]
    )
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "input_size3, segment_size3, hop_size3, normalize_segment_scale3, num_spk3",
    [
        (16000, None, None, False, 2),
        (35000, 2.4, 1.2, False, 2),
        (34000, 2.4, 1.2, True, 2),
    ],
)
def test_DiarSepSpeech3(
    config_file3,
    batch_size,
    input_size3,
    segment_size3,
    hop_size3,
    normalize_segment_scale3,
    num_spk3,
):
    diarize_speech = DiarSepSpeech(
        train_config=config_file3,
        segment_size=segment_size3,
        hop_size=hop_size3,
        normalize_segment_scale=normalize_segment_scale3,
        num_spk=num_spk3,
    )
    wav = torch.rand(batch_size, input_size3)
    diarize_speech(wav, fs=8000)
