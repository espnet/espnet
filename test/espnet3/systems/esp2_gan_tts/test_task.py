"""Tests for the ESPnet3 GAN-TTS task compatibility copy."""

import pytest

from espnet2.tasks.gan_tts import GANTTSTask as ESPnet2GANTTSTask
from espnet3.systems.esp2_gan_tts.task import GANTTSTask


def test_add_arguments():
    """Ensure parser construction succeeds."""
    GANTTSTask.get_parser()


def test_add_arguments_help():
    """Ensure parser help exits cleanly."""
    parser = GANTTSTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    """Ensure main help exits cleanly."""
    with pytest.raises(SystemExit):
        GANTTSTask.main(cmd=["--help"])


def test_main_print_config():
    """Ensure main print_config exits cleanly."""
    with pytest.raises(SystemExit):
        GANTTSTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    """Ensure main without args exits with usage."""
    with pytest.raises(SystemExit):
        GANTTSTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    """Ensure printed config can be parsed back."""
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        GANTTSTask.print_config(f)
    parser = GANTTSTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("inference", [True, False])
def test_data_names_match_espnet2(inference):
    """The copy must keep espnet2's required/optional data field names."""
    assert GANTTSTask.required_data_names(
        True, inference
    ) == ESPnet2GANTTSTask.required_data_names(True, inference)
    assert GANTTSTask.optional_data_names(
        True, inference
    ) == ESPnet2GANTTSTask.optional_data_names(True, inference)
    if not inference:
        assert "spembs" in GANTTSTask.optional_data_names(True, inference)
