import configargparse
import pytest

from espnet2.tasks.tts import TTSTask


@pytest.mark.parametrize("parser", [configargparse.ArgumentParser(), None])
def test_add_arguments(parser):
    TTSTask.get_parser(parser)


def test_add_arguments_help():
    parser = TTSTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        TTSTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        TTSTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        TTSTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        TTSTask.print_config(f)
    parser = TTSTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("name", TTSTask.feats_extract_choices())
def test_get_feats_extract_class(name):
    if name is not None:
        TTSTask.get_feats_extract_class(name)


@pytest.mark.parametrize("name", TTSTask.normalize_choices())
def test_get_normalize_class(name):
    if name is not None:
        TTSTask.get_normalize_class(name)


@pytest.mark.parametrize("name", TTSTask.tts_choices())
def test_get_tts_class(name):
    TTSTask.get_tts_class(name)
