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
    from espnet2.layers.batch_wav_augment import BatchWavAugment

    parser = DiarizationTask.get_parser()
    assert "wav_augment" not in parser.get_default("model_conf")
    args = parser.parse_args(
        [
            "--wav_augment",
            "batch_wav",
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
    assert isinstance(model.wav_augment, BatchWavAugment)
    # the noise list is read only on the first training step
    assert not (tmp_path / "noise.scp").exists()


def test_wav_augment_takes_the_frontend_sampling_rate(tmp_path):
    """Without wav_augment_conf fs, the noise is mixed at the frontend's fs."""
    parser = DiarizationTask.get_parser()
    common = ["--num_spk", "1", "--encoder_conf", "num_blocks=1"]
    args = parser.parse_args(
        common
        + [
            "--frontend_conf",
            "fs=8k",
            "--wav_augment",
            "batch_wav",
            "--wav_augment_conf",
            f"noise_scp={tmp_path / 'noise.scp'}",
        ]
    )
    assert DiarizationTask.build_model(args).wav_augment.fs == 8000

    args = parser.parse_args(
        common
        + [
            "--frontend_conf",
            "fs=8k",
            "--wav_augment",
            "batch_wav",
            "--wav_augment_conf",
            f"noise_scp={tmp_path / 'noise.scp'}",
            "--wav_augment_conf",
            "fs=16000",
        ]
    )
    with pytest.raises(ValueError):
        DiarizationTask.build_model(args)


def test_build_model_without_wav_augment():
    parser = DiarizationTask.get_parser()
    args = parser.parse_args(["--num_spk", "1", "--encoder_conf", "num_blocks=1"])
    assert DiarizationTask.build_model(args).wav_augment is None


@pytest.mark.parametrize(
    "overrides, ok",
    [
        (dict(iterator_type="chunk", chunk_left_context=400), True),
        (dict(iterator_type="sequence", chunk_left_context=400), False),
        (dict(iterator_type="chunk", chunk_left_context=0), False),
        (dict(iterator_type="chunk", chunk_left_context=300), False),
        (
            dict(
                iterator_type="chunk",
                chunk_left_context=400,
                valid_iterator_type="sequence",
            ),
            False,
        ),
        (
            dict(
                iterator_type="chunk",
                chunk_left_context=400,
                chunk_left_context_keys=["x"],
            ),
            False,
        ),
        (
            dict(iterator_type="sequence", chunk_left_context=0, collect_stats=True),
            True,
        ),
    ],
)
def test_wav_augment_context_needs_chunks_with_left_context(overrides, ok):
    """context > 0 trains only with chunks whose speech carries the context."""
    import argparse

    args = argparse.Namespace(
        iterator_type="chunk",
        valid_iterator_type=None,
        chunk_left_context=400,
        chunk_left_context_keys=["speech"],
        collect_stats=False,
    )
    vars(args).update(overrides)
    if ok:
        DiarizationTask._check_wav_augment_context(args, 400)
    else:
        with pytest.raises(ValueError, match="chunk_left_context"):
            DiarizationTask._check_wav_augment_context(args, 400)
    DiarizationTask._check_wav_augment_context(args, 0)  # no context: no check


@pytest.mark.parametrize(
    "left, default_fs, ok",
    [
        (800, 16000, True),  # 800 at 16 kHz = 400 at 8 kHz
        (400, 16000, False),  # 200 at 8 kHz: the model would crop 400
        (400, None, True),  # no default_fs: samples at the data's rate
        (801, 16000, False),  # not a whole number of samples at 8 kHz
    ],
)
def test_wav_augment_context_follows_chunk_default_fs(left, default_fs, ok):
    """--chunk_left_context is at --chunk_default_fs; the context at fs (8 kHz)."""
    import argparse

    args = argparse.Namespace(
        iterator_type="chunk",
        valid_iterator_type=None,
        chunk_left_context=left,
        chunk_left_context_keys=["speech"],
        chunk_default_fs=default_fs,
        collect_stats=False,
    )
    if ok:
        DiarizationTask._check_wav_augment_context(args, 400, 8000)
    else:
        with pytest.raises(ValueError, match="chunk_left_context"):
            DiarizationTask._check_wav_augment_context(args, 400, 8000)
