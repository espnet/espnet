from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch

from espnet2.bin.enh_inference import get_parser
from espnet2.bin.enh_inference import main
from espnet2.bin.enh_inference import SeparateSpeech
from espnet2.tasks.enh import EnhancementTask


EXAMPLE_ENH_EN_MODEL_ID = (
    "espnet/yen-ju-lu-dns_ins20_enh_train_enh_blstm_tf_raw_valid.loss.best"
)


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
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "input_size, segment_size, hop_size", [(16000, None, None), (35000, 2.4, 0.8)]
)
def test_SeparateSpeech(config_file, batch_size, input_size, segment_size, hop_size):
    separate_speech = SeparateSpeech(
        enh_train_config=config_file, segment_size=segment_size, hop_size=hop_size
    )
    wav = torch.rand(batch_size, input_size)
    separate_speech(wav, fs=8000)


@pytest.mark.execution_timeout(30)
def test_from_pretrained():
    try:
        from espnet_model_zoo.huggingface import from_huggingface  # NOQA
    except Exception:
        pytest.skip("No espnet_model_zoo found in your installation")
    separate_speech = SeparateSpeech.from_pretrained(EXAMPLE_ENH_EN_MODEL_ID)
    wav = torch.rand(1, 16000)
    separate_speech(wav, fs=8000)
