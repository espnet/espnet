from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.bin.spk_inference import Speech2Embedding, get_parser, main
from espnet2.tasks.spk import SpeakerTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def spk_config_file(tmp_path: Path):
    # Write default configuration file
    SpeakerTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "spk"),
            "--frontend",
            "asteroid_frontend",
            "--spk_num",
            "2",
        ]
    )
    return tmp_path / "spk" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_Speech2Embedding(spk_config_file):
    speech2embedding = Speech2Embedding(train_config=spk_config_file)
    speech = np.random.randn(1000)
    result = speech2embedding(speech)
    assert isinstance(result, torch.Tensor)
