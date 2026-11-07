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


def test_build_dual_mode_tcn_vad():
    """--encoder tcn with future taps and --model_conf look_aheads."""
    parser = DiarizationTask.get_parser()
    args = parser.parse_args(
        [
            "--num_spk",
            "1",
            "--encoder",
            "tcn",
            "--encoder_conf",
            "future_taps=2",
            "--model_conf",
            "look_aheads=[0,124]",
        ]
    )
    model = DiarizationTask.build_model(args)
    assert model.encoder.max_look_ahead == 124
    assert model.look_aheads == [0, 124]
