"""Tests for the ESPnet3 codec task wiring."""

import pytest

from espnet3.systems.esp2_codec.task import GANCodecTask


def test_add_arguments():
    """Ensure parser construction succeeds."""
    GANCodecTask.get_parser()


def test_main_print_config():
    """Ensure main print_config exits cleanly."""
    with pytest.raises(SystemExit):
        GANCodecTask.main(cmd=["--print_config"])


def test_required_data_names_are_audio_only():
    """The codec task reads one field, ``audio``, in training and inference."""
    assert GANCodecTask.required_data_names(train=True, inference=False) == ("audio",)
    assert GANCodecTask.required_data_names(train=False, inference=True) == ("audio",)
    assert GANCodecTask.optional_data_names() == ()


def test_default_config_names_the_codec_choices():
    """The copied task keeps the ESPnet2 codec choices and their default."""
    config = GANCodecTask.get_default_config()
    assert config["codec"] == "soundstream"
    assert "codec_conf" in config
