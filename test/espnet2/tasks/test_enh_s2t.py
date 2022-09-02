import pytest

from espnet2.tasks.enh_s2t import EnhS2TTask


def test_add_arguments():
    EnhS2TTask.get_parser()


def test_add_arguments_help():
    parser = EnhS2TTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        EnhS2TTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        EnhS2TTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        EnhS2TTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        EnhS2TTask.print_config(f)
    parser = EnhS2TTask.get_parser()
    parser.parse_args(["--config", str(config_file)])
