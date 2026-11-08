import numpy as np
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


def _build_preprocess(extra):
    args = DiarizationTask.get_parser().parse_args(extra)
    return DiarizationTask.build_preprocess_fn(args, train=True)


@pytest.mark.parametrize("n_cols", [0, 1, 2, 3])
def test_spk_labels_padded_to_num_spk(n_cols):
    preprocess = _build_preprocess(["--num_spk", "2"])
    labels = np.ones((100, n_cols), dtype=np.float32)
    out = preprocess("uid", {"speech": np.zeros(100), "spk_labels": labels})
    want = max(n_cols, 2)  # more columns than num_spk stay as they are
    assert out["spk_labels"].shape == (100, want)
    assert out["spk_labels"][:, :n_cols].all()
    assert not out["spk_labels"][:, n_cols:].any()


def test_spk_labels_of_eda_not_padded():
    preprocess = _build_preprocess(["--num_spk", "2", "--attractor", "rnn"])
    labels = np.ones((100, 1), dtype=np.float32)
    out = preprocess("uid", {"speech": np.zeros(100), "spk_labels": labels})
    assert out["spk_labels"].shape == (100, 1)


def test_no_preprocessor():
    assert _build_preprocess(["--use_preprocessor", "false"]) is None
