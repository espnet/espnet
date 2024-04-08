import pytest

from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask


def test_add_arguments():
    TargetSpeakerExtractionTask.get_parser()


def test_add_arguments_help():
    parser = TargetSpeakerExtractionTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        TargetSpeakerExtractionTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        TargetSpeakerExtractionTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        TargetSpeakerExtractionTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        TargetSpeakerExtractionTask.print_config(f)
    parser = TargetSpeakerExtractionTask.get_parser()
    parser.parse_args(["--config", str(config_file)])
