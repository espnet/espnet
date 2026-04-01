import pytest

pytest.importorskip("whisper")

from espnet2.tasks.sot_asr import SOTASRTask


def test_add_arguments():
    SOTASRTask.get_parser()


def test_add_arguments_help():
    parser = SOTASRTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        SOTASRTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        SOTASRTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        SOTASRTask.main(cmd=[])
