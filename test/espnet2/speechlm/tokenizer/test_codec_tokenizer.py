import pytest
import torch
from pathlib import Path
from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer


@pytest.fixture()
def tokenizer_config_yaml(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
        encoder_layers: 2
        quant_n: 16
        """
    )
    return config_file


def test_codec_tokenizer_init(tokenizer_config_yaml):
    tokenizer = CodecTokenizer(
        codec_choice="beats", codec_fs=16000, config_path=tokenizer_config_yaml
    )
    assert tokenizer is not None


def test_encode(tokenizer_config_yaml):
    tokenizer = CodecTokenizer(
        codec_choice="beats", codec_fs=16000, config_path=tokenizer_config_yaml
    )
    wav_in = torch.randn(2, 1, 16000)
    output = tokenizer.encode(wav_in)
    assert output is not None
