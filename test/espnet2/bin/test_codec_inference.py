from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch
import yaml

from espnet2.bin.gan_codec_inference import AudioCoding, get_parser, main
from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel
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


@pytest.fixture()
def funcodec_config_file(tmp_path: Path):
    codec_conf = dict(
        sampling_rate=120,
        generator_params=dict(
            hidden_dim=2,
            codebook_dim=2,
            encdec_channels=3,
            encdec_n_filters=32,
            encdec_ratios=[[4, 1], [2, 1]],
            encdec_kernel_size=1,
            encdec_residual_kernel_size=1,
            encdec_last_kernel_size=1,
            quantizer_n_q=2,
            quantizer_bins=4,
            quantizer_kmeans_init=False,
            quantizer_target_bandwidth=[15],
            codec_domain=["mag_phase", "mag_phase"],
            domain_conf={"n_fft": 16, "hop_length": 4},
            audio_normalize=True,
        ),
    )
    config = tmp_path / "funcodec.yaml"
    config.write_text(yaml.safe_dump(dict(codec="funcodec", codec_conf=codec_conf)))
    GANCodecTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--config",
            str(config),
            "--output_dir",
            str(tmp_path / "funcodec"),
        ]
    )
    return tmp_path / "funcodec" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_AudioCoding_funcodec_audio_normalize_restores_volume(funcodec_config_file):
    torch.manual_seed(0)
    audio_coding = AudioCoding(train_config=funcodec_config_file)
    wav = torch.randn(128)
    resyn = audio_coding(wav)["resyn_audio"]
    resyn_loud = audio_coding(100 * wav)["resyn_audio"]
    assert resyn.shape == wav.shape
    torch.testing.assert_close(resyn_loud, 100 * resyn, rtol=1e-4, atol=1e-5)


class EncodeDecodeOnlyCodec(AbsGANCodec):
    """Codec that only implements the methods required by AbsGANCodec."""

    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Identity()
        self.discriminator = torch.nn.Identity()

    def meta_info(self):
        return {"fs": 16000}

    def forward(self, audio, forward_generator=True, **kwargs):
        raise NotImplementedError

    def encode(self, x, **kwargs):
        # (B, 1, T) -> (N_stream=1, B, T)
        return (x * 10).round().long().transpose(0, 1)

    def decode(self, x, **kwargs):
        return x.transpose(0, 1).float() / 10


@pytest.mark.execution_timeout(5)
def test_AudioCoding_codec_without_inference(config_file):
    audio_coding = AudioCoding(train_config=config_file)
    audio_coding.model = ESPnetGANCodecModel(EncodeDecodeOnlyCodec())
    wav = torch.rand(160)
    output_dict = audio_coding(wav)
    codes = audio_coding.model.encode(wav)
    torch.testing.assert_close(output_dict["codes"], codes)
    torch.testing.assert_close(
        output_dict["resyn_audio"], audio_coding.model.decode(codes).view(-1)
    )
