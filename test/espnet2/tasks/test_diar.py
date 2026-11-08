import pytest

from espnet2.tasks.diar import DiarizationTask


def test_add_arguments():
    DiarizationTask.get_parser()


def test_add_arguments_help():
    parser = DiarizationTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        DiarizationTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        DiarizationTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        DiarizationTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        DiarizationTask.print_config(f)
    parser = DiarizationTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


def test_preprocessor_receives_augmentation_options(tmp_path):
    noise = tmp_path / "noise.scp"
    noise.write_text("n1 /dev/null\n")
    rir = tmp_path / "rir.scp"
    rir.write_text("r1 /dev/null\n")
    parser = DiarizationTask.get_parser()
    args = parser.parse_args(
        [
            "--noise_scp",
            str(noise),
            "--rir_scp",
            str(rir),
            "--noise_db_range=-5_20",
        ]
    )
    train_fn = DiarizationTask.build_preprocess_fn(args, train=True)
    assert train_fn.noises == ["/dev/null"]
    assert train_fn.rirs == ["/dev/null"]
    assert train_fn.rir_remove_delay is True
    assert (train_fn.noise_db_low, train_fn.noise_db_high) == (-5.0, 20.0)
    # augmentation is for training only
    valid_fn = DiarizationTask.build_preprocess_fn(args, train=False)
    assert valid_fn.noises is None and valid_fn.rirs is None
