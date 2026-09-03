from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch

from espnet2.bin.gan_codec_inference import AudioCoding, get_parser, main
from espnet2.tasks.gan_codec import GANCodecTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def config_file(tmp_path: Path):
    # Write default configuration file
    GANCodecTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "gan_codec"),
        ]
    )
    return tmp_path / "gan_codec" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_AudioCoding(config_file):
    audio_coding = AudioCoding(train_config=config_file)
    wav = torch.rand(1, 16000)
    output_dict = audio_coding(wav)
    assert "resyn_audio" in output_dict
    assert "codes" in output_dict
    assert isinstance(output_dict["resyn_audio"], torch.Tensor)
    assert isinstance(output_dict["codes"], torch.Tensor)


@pytest.mark.execution_timeout(5)
def test_AudioCoding_decode(config_file):
    audio_coding = AudioCoding(train_config=config_file)
    wav = torch.rand(1, 16000)
    encode_dict = audio_coding(wav, encode_only=True)
    assert "codes" in encode_dict
    assert "resyn_audio" not in encode_dict
    assert isinstance(encode_dict["codes"], torch.Tensor)

    decode_dict = audio_coding.decode(encode_dict["codes"])
    assert "resyn_audio" in decode_dict
    assert isinstance(decode_dict["resyn_audio"], torch.Tensor)
