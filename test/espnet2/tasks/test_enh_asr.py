import pytest

from espnet2.tasks.enh_asr import EnhASRTask


def test_add_arguments():
    EnhASRTask.get_parser()


def test_add_arguments_help():
    parser = EnhASRTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        EnhASRTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        EnhASRTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        EnhASRTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        EnhASRTask.print_config(f)
    parser = EnhASRTask.get_parser()
    parser.parse_args(["--config", str(config_file)])
