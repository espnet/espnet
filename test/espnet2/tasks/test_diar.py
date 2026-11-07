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


def test_build_model_with_wav_augment(tmp_path):
    """--wav_augment builds the module into the model; it is not a model_conf."""
    from espnet2.layers.batch_noise_augment import BatchNoiseAugment

    parser = DiarizationTask.get_parser()
    assert "wav_augment" not in parser.get_default("model_conf")
    args = parser.parse_args(
        [
            "--wav_augment",
            "batch_noise",
            "--wav_augment_conf",
            f"noise_scp={tmp_path / 'noise.scp'}",
            "--num_spk",
            "1",
            "--encoder",
            "transformer",
            "--encoder_conf",
            "num_blocks=1",
        ]
    )
    model = DiarizationTask.build_model(args)
    assert isinstance(model.wav_augment, BatchNoiseAugment)
    # the noise list is read only on the first training step
    assert not (tmp_path / "noise.scp").exists()
