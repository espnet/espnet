import pytest

from espnet2.tasks.hubert import HubertTask


def test_add_arguments():
    HubertTask.get_parser()


def test_add_arguments_help():
    parser = HubertTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        HubertTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        HubertTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        HubertTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        HubertTask.print_config(f)
    parser = HubertTask.get_parser()
    parser.parse_args(["--config", str(config_file)])
