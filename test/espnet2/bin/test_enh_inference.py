from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch

from espnet2.bin.enh_inference import get_parser
from espnet2.bin.enh_inference import main
from espnet2.bin.enh_inference import SeparateSpeech
from espnet2.tasks.enh import EnhancementTask


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
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_SeparateSpeech(config_file):
    separate_speech = SeparateSpeech(enh_train_config=config_file)
    wav = torch.rand(1, 16000)
    separate_speech(wav, fs=16000)
