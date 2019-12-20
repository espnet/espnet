import configargparse
import pytest

from espnet2.tasks.lm import LMTask


@pytest.mark.parametrize("parser", [configargparse.ArgumentParser(), None])
def test_add_arguments(parser):
    LMTask.get_parser(parser)


def test_add_arguments_help():
    parser = LMTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        LMTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        LMTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        LMTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        LMTask.print_config(f)
    parser = LMTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("name", LMTask.lm_choices())
def test_get_lm_class(name):
    LMTask.get_lm_class(name)
