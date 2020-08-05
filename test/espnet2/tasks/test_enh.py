import pytest

from espnet2.tasks.enh import EnhancementTask


def test_add_arguments():
    EnhancementTask.get_parser()


def test_add_arguments_help():
    parser = EnhancementTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        EnhancementTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        EnhancementTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        EnhancementTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        EnhancementTask.print_config(f)
    parser = EnhancementTask.get_parser()
    parser.parse_args(["--config", str(config_file)])
