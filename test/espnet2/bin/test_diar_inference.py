from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch

from espnet2.bin.diar_inference import DiarizeSpeech
from espnet2.bin.diar_inference import get_parser
from espnet2.bin.diar_inference import main
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.enh_s2t import EnhS2TTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def diar_config_file(tmp_path: Path):
    # Write default configuration file
    DiarizationTask.main(
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
    "input_size, segment_size, normalize_segment_scale, num_spk",
    [(16000, None, False, 2), (35000, 2.4, False, 2), (34000, 2.4, True, 2)],
)
def test_DiarizeSpeech(
    diar_config_file,
    batch_size,
    input_size,
    segment_size,
    normalize_segment_scale,
    num_spk,
):
    diarize_speech = DiarizeSpeech(
        train_config=diar_config_file,
        segment_size=segment_size,
        normalize_segment_scale=normalize_segment_scale,
        num_spk=num_spk,
    )
    wav = torch.rand(batch_size, input_size)
    diarize_speech(wav, fs=8000)


@pytest.fixture()
def diar_config_file2(tmp_path: Path):
    # Write default configuration file
    DiarizationTask.main(
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
    "input_size, segment_size, normalize_segment_scale",
    [
        (16000, None, False),
        (35000, 2.4, False),
        (34000, 2.4, True),
    ],
)
@pytest.mark.parametrize("num_spk", [None, 2])
def test_DiarizeSpeech2(
    diar_config_file2,
    batch_size,
    input_size,
    segment_size,
    normalize_segment_scale,
    num_spk,
):
    diarize_speech = DiarizeSpeech(
        train_config=diar_config_file2,
        segment_size=segment_size,
        normalize_segment_scale=normalize_segment_scale,
        num_spk=num_spk,
    )
    wav = torch.rand(batch_size, input_size)
    diarize_speech(wav, fs=8000)


@pytest.fixture()
def diarsep_config_file(tmp_path: Path):
    # Write default configuration file
    EnhS2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--diar_num_spk",
            "2",
            "--diar_frontend",
            "default",
            "--subtask_series",
            "enh",
            "diar",
            "--diar_input_size",
            "128",
            "--enh_separator",
            "tcn_nomask",
            "--enh_separator_conf",
            "bottleneck_dim=128",
            "--enh_mask_module",
            "multi_mask",
            "--enh_mask_module_conf",
            "bottleneck_dim=128",
        ]
    )
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "input_size, segment_size, hop_size, normalize_segment_scale, num_spk",
    [
        (16000, None, None, False, 2),
        (35000, 2.4, 1.2, False, 2),
        (34000, 2.4, 1.2, True, 2),
    ],
)
@pytest.mark.parametrize("multiply_diar_result", [True, False])
def test_DiarSepSpeech(
    diarsep_config_file,
    batch_size,
    input_size,
    segment_size,
    hop_size,
    normalize_segment_scale,
    num_spk,
    multiply_diar_result,
):
    diarize_speech = DiarizeSpeech(
        train_config=diarsep_config_file,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        num_spk=num_spk,
        enh_s2t_task=True,
        normalize_output_wav=True,
        multiply_diar_result=multiply_diar_result,
    )
    wav = torch.rand(batch_size, input_size)
    diarize_speech(wav, fs=8000)
