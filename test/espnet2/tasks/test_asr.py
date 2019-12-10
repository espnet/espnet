import configargparse
import pytest

from espnet2.tasks.asr import ASRTask


@pytest.mark.parametrize("parser", [configargparse.ArgumentParser(), None])
def test_add_arguments(parser):
    ASRTask.add_arguments(parser)


def test_add_arguments_help():
    parser = ASRTask.add_arguments()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        ASRTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        ASRTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        ASRTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        ASRTask.print_config(f)
    parser = ASRTask.add_arguments()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("name", ASRTask.frontend_choices())
def test_get_frontend_class(name):
    if name is not None:
        ASRTask.get_frontend_class(name)


@pytest.mark.parametrize("name", ASRTask.normalize_choices())
def test_get_normalize_class(name):
    if name is not None:
        ASRTask.get_normalize_class(name)


@pytest.mark.parametrize("name", ASRTask.encoder_decoder_choices())
def test_get_encoder_decoder_class(name):
    ASRTask.get_encoder_decoder_class(name)
