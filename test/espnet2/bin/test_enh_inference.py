from argparse import ArgumentParser
from pathlib import Path
import string

import pytest
import torch

from espnet2.bin.enh_inference import get_parser
from espnet2.bin.enh_inference import main
from espnet2.bin.enh_inference import SeparateSpeech
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_s2t import EnhS2TTask


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
        ]
    )
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
