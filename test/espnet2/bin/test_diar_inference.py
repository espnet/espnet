from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.bin.diar_inference import DiarizeSpeech, get_parser, main
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.enh_s2t import EnhS2TTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def diar_config_file(tmp_path: Path):
    # Write default configuration file
    DiarizationTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--num_spk",
            "2",
        ]
    )
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "input_size, segment_size, normalize_segment_scale, num_spk",
    [(16000, None, False, 2), (35000, 2.4, False, 2), (34000, 2.4, True, 2)],
)
def test_DiarizeSpeech(
    diar_config_file,
    batch_size,
    input_size,
    segment_size,
    normalize_segment_scale,
    num_spk,
):
    diarize_speech = DiarizeSpeech(
        train_config=diar_config_file,
        segment_size=segment_size,
        normalize_segment_scale=normalize_segment_scale,
        num_spk=num_spk,
    )
    wav = torch.rand(batch_size, input_size)
    waves, spk_prediction = diarize_speech(wav, fs=8000)
    # a diarization model returns no waveforms and the speaker activities
    assert waves is None
    assert isinstance(spk_prediction, np.ndarray), type(spk_prediction)
    assert spk_prediction.shape[0] == batch_size
    if num_spk is not None:
        assert spk_prediction.shape[2] == num_spk


@pytest.fixture()
def diar_config_file2(tmp_path: Path):
    # Write default configuration file
    DiarizationTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--attractor",
            "rnn",
            "--attractor_conf",
            "unit=256",
            "--num_spk",
            "2",
        ]
    )
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "input_size, segment_size, normalize_segment_scale",
    [
        (16000, None, False),
        (35000, 2.4, False),
        (34000, 2.4, True),
    ],
)
@pytest.mark.parametrize("num_spk", [None, 2])
def test_DiarizeSpeech2(
    diar_config_file2,
    batch_size,
    input_size,
    segment_size,
    normalize_segment_scale,
    num_spk,
):
    diarize_speech = DiarizeSpeech(
        train_config=diar_config_file2,
        segment_size=segment_size,
        normalize_segment_scale=normalize_segment_scale,
        num_spk=num_spk,
    )
    wav = torch.rand(batch_size, input_size)
    waves, spk_prediction = diarize_speech(wav, fs=8000)
    # a diarization model returns no waveforms and the speaker activities
    assert waves is None
    assert isinstance(spk_prediction, np.ndarray), type(spk_prediction)
    assert spk_prediction.shape[0] == batch_size
    if num_spk is not None:
        assert spk_prediction.shape[2] == num_spk


@pytest.fixture()
def diarsep_config_file(tmp_path: Path):
    # Write default configuration file
    EnhS2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--diar_num_spk",
            "2",
            "--diar_frontend",
            "default",
            "--subtask_series",
            "enh",
            "diar",
            "--diar_input_size",
            "128",
            "--enh_separator",
            "tcn_nomask",
            "--enh_separator_conf",
            "bottleneck_dim=128",
            "--enh_mask_module",
            "multi_mask",
            "--enh_mask_module_conf",
            "bottleneck_dim=128",
        ]
    )
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "input_size, segment_size, hop_size, normalize_segment_scale, num_spk",
    [
        (16000, None, None, False, 2),
        (35000, 2.4, 1.2, False, 2),
        (34000, 2.4, 1.2, True, 2),
    ],
)
@pytest.mark.parametrize("multiply_diar_result", [True, False])
def test_DiarSepSpeech(
    diarsep_config_file,
    batch_size,
    input_size,
    segment_size,
    hop_size,
    normalize_segment_scale,
    num_spk,
    multiply_diar_result,
):
    diarize_speech = DiarizeSpeech(
        train_config=diarsep_config_file,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        num_spk=num_spk,
        enh_s2t_task=True,
        normalize_output_wav=True,
        multiply_diar_result=multiply_diar_result,
    )
    wav = torch.rand(batch_size, input_size)
    waves, spk_prediction = diarize_speech(wav, fs=8000)
    # enh + diar returns the separated waveforms and the speaker activities
    assert len(waves) == spk_prediction.shape[2]
    assert spk_prediction.shape[0] == batch_size


def _write_config(tmp_path, *args):
    DiarizationTask.main(
        cmd=["--dry_run", "true", "--output_dir", str(tmp_path), "--num_spk", "2"]
        + list(args)
    )
    return tmp_path / "config.yaml"


@pytest.mark.parametrize(
    "model_args, num_spk",
    [
        ((), 2),  # EEND, transformer encoder, center=True STFT
        (("--frontend_conf", "center=false"), 2),
        (("--encoder", "rnn"), 2),
        (("--attractor", "rnn", "--attractor_conf", "unit=256"), 2),
        (("--attractor", "rnn", "--attractor_conf", "unit=256"), None),
    ],
)
def test_diarize_batch_equals_one_by_one(tmp_path, model_args, num_spk):
    """A padded batch gives each utterance what it gets alone."""
    torch.manual_seed(0)
    diarize_speech = DiarizeSpeech(
        train_config=_write_config(tmp_path, *model_args), num_spk=num_spk
    )
    lengths = [16000, 12345, 9001, 16000]
    wavs = [torch.randn(n) * 0.1 for n in lengths]
    batch = torch.zeros(len(wavs), max(lengths))
    for b, w in enumerate(wavs):
        batch[b, : len(w)] = w

    batched = diarize_speech.diarize_batch(batch, torch.tensor(lengths), fs=8000)
    assert len(batched) == len(wavs)
    for b, w in enumerate(wavs):
        _, alone = diarize_speech(w[None], fs=8000)
        alone = alone[0]
        assert batched[b].shape == alone.shape, (b, batched[b].shape, alone.shape)
        np.testing.assert_allclose(batched[b], alone, rtol=1e-5, atol=1e-6)


def test_diarize_batch_without_lengths_is_the_unpadded_batch(tmp_path):
    torch.manual_seed(0)
    diarize_speech = DiarizeSpeech(train_config=_write_config(tmp_path), num_spk=2)
    batch = torch.randn(3, 12000) * 0.1
    batched = diarize_speech.diarize_batch(batch, fs=8000)
    _, together = diarize_speech(batch, fs=8000)
    for b in range(3):
        np.testing.assert_allclose(batched[b], together[b], rtol=1e-5, atol=1e-6)


def test_diarize_batch_segment_wise_runs_one_by_one(tmp_path):
    """Segment-wise diarization: each utterance as __call__ gives it alone."""
    torch.manual_seed(0)
    diarize_speech = DiarizeSpeech(
        train_config=_write_config(tmp_path), num_spk=2, segment_size=1.0
    )
    lengths = [20000, 7000]
    batch = torch.zeros(2, 20000)
    wavs = [torch.randn(n) * 0.1 for n in lengths]
    for b, w in enumerate(wavs):
        batch[b, : len(w)] = w
    batched = diarize_speech.diarize_batch(batch, torch.tensor(lengths), fs=8000)
    for b, w in enumerate(wavs):
        np.testing.assert_allclose(
            batched[b], diarize_speech(w[None], fs=8000)[1][0], rtol=1e-5, atol=1e-6
        )


def test_diarize_batch_estimates_the_number_of_speakers_per_utterance(
    tmp_path, monkeypatch
):
    """EEND-EDA without num_spk: each utterance keeps its own speaker count."""
    torch.manual_seed(0)
    diarize_speech = DiarizeSpeech(
        train_config=_write_config(
            tmp_path, "--attractor", "rnn", "--attractor_conf", "unit=256"
        ),
        num_spk=None,
    )
    attractor = diarize_speech.diar_model.attractor
    forward = attractor.forward
    first_negative = {16000: 3, 9001: 1}  # utterance length -> number of speakers

    def fake_attractor(enc, lens, dec):
        att, _ = forward(enc, lens, dec)
        prob = torch.ones(enc.size(0), dec.size(1), 1)
        for b, n in enumerate(lens.tolist()):
            k = 3 if n == lens.max() else 1
            prob[b, k:] = -1.0
        return att, prob

    monkeypatch.setattr(attractor, "forward", fake_attractor)
    lengths = [9001, 16000]
    batch = torch.randn(2, 16000) * 0.1
    batch[0, 9001:] = 0
    out = diarize_speech.diarize_batch(batch, torch.tensor(lengths), fs=8000)
    assert out[0].shape[1] == first_negative[9001]
    assert out[1].shape[1] == first_negative[16000]
    assert out[0].shape[0] < out[1].shape[0]
