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
    "input_size, segment_size, normalize_segment_scale, num_spk",
    [(16000, None, False, 2), (35000, 2.4, False, 2), (34000, 2.4, True, 2)],
)
def test_DiarSepSpeech(
    config_file, batch_size, input_size, segment_size, normalize_segment_scale, num_spk
):
    diarize_speech = DiarSepSpeech(
        train_config=config_file,
        segment_size=segment_size,
        normalize_segment_scale=normalize_segment_scale,
        num_spk=num_spk,
    )
    wav = torch.rand(batch_size, input_size)
    diarize_speech(wav, fs=8000)
