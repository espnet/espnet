import configargparse
import pytest

from espnet2.tasks.abs_task import AbsTask


@pytest.mark.parametrize("parser", [configargparse.ArgumentParser(), None])
def test_add_arguments(parser):
    AbsTask.add_arguments(parser)


def test_add_arguments_help():
    parser = AbsTask.add_arguments()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        AbsTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        AbsTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        AbsTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        AbsTask.print_config(f)
    parser = AbsTask.add_arguments()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize("name", AbsTask.optimizer_choices())
def test_get_optimizer_class(name):
    AbsTask.get_optimizer_class(name)


@pytest.mark.parametrize("name", AbsTask.epoch_scheduler_choices())
def test_get_epoch_scheduler_class(name):
    if name is not None:
        AbsTask.get_epoch_scheduler_class(name)


@pytest.mark.parametrize("name", AbsTask.batch_scheduler_choices())
def test_get_batch_scheduler_class(name):
    if name is not None:
        AbsTask.get_batch_scheduler_class(name)
