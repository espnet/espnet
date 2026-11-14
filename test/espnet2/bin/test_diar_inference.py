from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
import torch

from espnet2.bin.diar_inference import (
    VAD_MEDIAN,
    VAD_THRESHOLD,
    DiarizeSpeech,
    Segment,
    VoiceActivityDetector,
    format_segments,
    frames_to_segments,
    get_parser,
    main,
    read_audio,
)
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
        # TCN: causal, and with a look-ahead (its convolutions also see later
        # frames, so the padding of a shorter utterance must not reach its own
        # frames): part of the maximum (124 frames) and all of it
        (("--encoder", "tcn", "--encoder_conf", "future_taps=0"), 2),
        (("--encoder", "tcn", "--encoder_conf", "future_taps=2"), 2),
        (
            ("--encoder", "tcn")
            + ("--encoder_conf", "future_taps=2", "--encoder_conf", "look_ahead=14"),
            2,
        ),
        (
            ("--encoder", "tcn")
            + ("--encoder_conf", "future_taps=2", "--encoder_conf", "look_ahead=124"),
            2,
        ),
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


@pytest.mark.parametrize("gap", [1, 127, 255, 256, 257])
def test_diarize_batch_short_gap_center_stft(tmp_path, gap):
    """An utterance a little shorter than the longest one (gap < n_fft / 2).

    With a centred STFT (default n_fft=512, hop 128) the utterance alone is
    padded by the reflection of its own last 256 samples; in the batch it
    must get the same frames.
    """
    torch.manual_seed(0)
    diarize_speech = DiarizeSpeech(train_config=_write_config(tmp_path), num_spk=2)
    lengths = [16000, 16000 - gap]
    wavs = [torch.randn(n) * 0.1 for n in lengths]
    batch = torch.zeros(2, 16000)
    for b, w in enumerate(wavs):
        batch[b, : len(w)] = w
    batched = diarize_speech.diarize_batch(batch, torch.tensor(lengths), fs=8000)
    for b, w in enumerate(wavs):
        alone = diarize_speech(w[None], fs=8000)[1][0]
        assert batched[b].shape == alone.shape
        np.testing.assert_allclose(batched[b], alone, rtol=1e-5, atol=1e-6)


def test_diarize_batch_multichannel_runs_one_by_one(tmp_path):
    """Multi-channel input (B, T, C): each utterance as __call__ gives it alone."""
    torch.manual_seed(0)
    diarize_speech = DiarizeSpeech(train_config=_write_config(tmp_path), num_spk=2)
    lengths = [12000, 9000]
    batch = torch.zeros(2, 12000, 2)
    wavs = [torch.randn(n, 2) * 0.1 for n in lengths]
    for b, w in enumerate(wavs):
        batch[b, : len(w)] = w
    batched = diarize_speech.diarize_batch(batch, torch.tensor(lengths), fs=8000)
    for b, w in enumerate(wavs):
        alone = diarize_speech(w[None], fs=8000)[1][0]
        np.testing.assert_allclose(batched[b], alone, rtol=1e-5, atol=1e-6)


def test_diarize_batch_utterance_shorter_than_half_window(tmp_path):
    """An utterance of n_fft / 2 samples or fewer: same outcome as __call__."""
    torch.manual_seed(0)
    diarize_speech = DiarizeSpeech(train_config=_write_config(tmp_path), num_spk=2)
    lengths = [4000, 200]
    batch = torch.zeros(2, 4000)
    wavs = [torch.randn(n) * 0.1 for n in lengths]
    for b, w in enumerate(wavs):
        batch[b, : len(w)] = w
    try:
        expected = [diarize_speech(w[None], fs=8000)[1][0] for w in wavs]
    except RuntimeError:
        # torch.stft cannot reflect it alone either
        with pytest.raises(RuntimeError):
            diarize_speech.diarize_batch(batch, torch.tensor(lengths), fs=8000)
        return
    batched = diarize_speech.diarize_batch(batch, torch.tensor(lengths), fs=8000)
    for b in range(2):
        np.testing.assert_allclose(batched[b], expected[b], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("trained", [0, 14])
def test_look_ahead_is_set_at_inference(tmp_path, trained):
    """--look_ahead sets the encoder's look-ahead; None keeps the trained one."""
    DiarizationTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--num_spk",
            "1",
            "--encoder",
            "tcn",
            "--encoder_conf",
            "future_taps=2",
            "--encoder_conf",
            f"look_ahead={trained}",
        ]
    )
    config = tmp_path / "config.yaml"
    kept = DiarizeSpeech(train_config=config).diar_model.encoder.look_ahead
    assert kept == trained
    offline = DiarizeSpeech(train_config=config, look_ahead=124, num_spk=1)
    assert offline.diar_model.encoder.look_ahead == 124
    _, prediction = offline(torch.randn(1, 8000), fs=8000)
    assert prediction.shape[-1] == 1


def test_look_ahead_needs_a_look_ahead_encoder(diar_config_file):
    with pytest.raises(ValueError):
        DiarizeSpeech(train_config=diar_config_file, look_ahead=4)


# =============================================================================
# Voice activity detection: VoiceActivityDetector, the rule, files and batches.
# The models are small and randomly initialised, built the way the
# frontia_vad_v1 recipe builds its models (diar1, num_spk 1, TCN, STFT without
# centring), so nothing is downloaded; what is checked is that the ways of
# running a model agree, not what it detects. Streaming is in
# test_diar_inference_streaming.py.
# =============================================================================

FS = 16000


def _model(tmp_path, center=False, num_blocks=3):
    DiarizationTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--num_spk",
            "1",
            "--frontend_conf",
            f"fs={FS}",
            "--frontend_conf",
            "n_fft=512",
            "--frontend_conf",
            "hop_length=128",
            "--frontend_conf",
            f"center={str(center).lower()}",
            "--normalize",
            "none",
            "--encoder",
            "tcn",
            "--encoder_conf",
            "causal=true",
            "--encoder_conf",
            f"num_blocks={num_blocks}",
            "--encoder_conf",
            "num_repeats=1",
            "--encoder_conf",
            "output_size=8",
        ]
    )
    torch.manual_seed(0)
    return VoiceActivityDetector(
        DiarizeSpeech(train_config=tmp_path / "config.yaml", num_spk=1)
    )


def _speech(seconds, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=int(seconds * FS)) * 0.1).astype(np.float32)


class _LookAheadEncoder:
    max_look_ahead = 6
    receptive_field = 10

    def __init__(self):
        self.set = []

    def set_look_ahead(self, frames):
        self.set.append(frames)


def _stub_detector():
    """A detector over a stand-in DiarizeSpeech whose encoder looks ahead."""
    encoder = _LookAheadEncoder()

    def diarize(speech, fs):
        frames = (speech.shape[1] - 512) // 128 + 1
        return None, np.full((1, frames, 1), 0.9, dtype=np.float32)

    diarize.diar_train_args = SimpleNamespace(
        frontend_conf={"fs": "16k", "n_fft": 512, "hop_length": 128, "center": False}
    )
    diarize.diar_model = SimpleNamespace(encoder=encoder)
    return VoiceActivityDetector(diarize), encoder


def test_segments_follow_make_rttm():
    """Frames above the threshold, as [start, end) in seconds."""
    probs = np.array([0.1, 0.6, 0.9, 0.5, 0.2, 0.8, 0.8])
    # 0.5 is not above 0.5, as in make_rttm.py (data > threshold)
    assert frames_to_segments(probs, 0.01, 0.5, 1) == [
        Segment(0.01, 0.03),
        Segment(0.05, 0.07),
    ]


def test_median_filter_removes_a_blip_and_fills_a_gap():
    probs = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1], dtype=float)
    # the lone frame at 2 goes; the gap at 7 is filled
    assert frames_to_segments(probs, 1.0, 0.5, 3) == [Segment(5.0, 11.0)]


def test_no_speech_gives_no_segments():
    assert frames_to_segments(np.zeros(10), 0.01) == []
    assert frames_to_segments(np.zeros(0), 0.01) == []


@pytest.mark.parametrize("median", [0, 2, -1])
def test_median_must_be_odd_and_positive(median):
    with pytest.raises(ValueError):
        frames_to_segments(np.zeros(5), 0.01, median=median)


def test_each_published_model_has_a_valid_rule():
    from espnet2.bin.diar_inference import VAD_MODELS, VAD_RULES

    assert set(VAD_MODELS) == set(VAD_RULES) == {"online", "offline"}
    for threshold, median in VAD_RULES.values():
        assert 0 < threshold < 1 and median >= 1 and median % 2 == 1
    assert (VAD_THRESHOLD, VAD_MEDIAN) == VAD_RULES["offline"]
    assert all(tag.startswith("espnet/") for tag in VAD_MODELS.values())


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_for_mode_loads_the_model_of_the_mode_with_its_rule(monkeypatch, mode):
    from espnet2.bin import diar_inference as vad_module

    loaded = {}

    def load(tag, device="cpu", threshold=None, median=None):
        loaded.update(tag=tag, threshold=threshold, median=median)
        return "detector"

    monkeypatch.setattr(VoiceActivityDetector, "load", staticmethod(load))
    monkeypatch.setitem(vad_module.VAD_RULES, mode, (0.35, 7))
    assert VoiceActivityDetector.for_mode(mode) == "detector"
    assert loaded == {
        "tag": vad_module.VAD_MODELS[mode],
        "threshold": 0.35,
        "median": 7,
    }
    VoiceActivityDetector.for_mode(mode, "local.zip")
    assert loaded["tag"] == "local.zip"
    with pytest.raises(ValueError):
        VoiceActivityDetector.for_mode("sideways")


def test_format_segments_is_tab_separated():
    text = format_segments([Segment(0.0, 1.234), Segment(2.5, 3.0)])
    assert text == "0.00\t1.23\n2.50\t3.00"
    assert Segment(1.0, 3.5).duration == 2.5


def test_read_audio_takes_the_first_channel_and_resamples(tmp_path):
    stereo = np.stack([np.ones(800), -np.ones(800)], axis=1).astype(np.float32) * 0.5
    soundfile.write(tmp_path / "a.wav", stereo, 8000)
    out = read_audio(tmp_path / "a.wav", 16000)
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) - 1600) <= 1
    assert out[400:1200].mean() > 0.4  # channel 0, not the average


def test_reads_the_rate_and_frames_from_the_model(tmp_path):
    vad = _model(tmp_path)
    assert (vad.fs, vad.n_fft, vad.hop, vad.center) == (FS, 512, 128, False)
    assert vad.frame_shift == 128 / FS
    assert vad.receptive_field > 1
    # a causal TCN reads no future frame: offline is online
    assert vad.max_look_ahead == 0 and vad.offline_look_ahead == 0


def test_frames_of_a_recording(tmp_path):
    vad = _model(tmp_path)
    probs = vad.probabilities(_speech(1.0))
    assert probs.shape == ((FS - 512) // 128 + 1,)
    assert np.all((probs >= 0) & (probs <= 1))
    # shorter than one frame
    assert vad.probabilities(np.zeros(100, dtype=np.float32)).shape == (0,)


def test_batch_equals_each_recording_alone(tmp_path):
    vad = _model(tmp_path)
    speeches = [_speech(s, i) for i, s in enumerate((2.0, 0.7, 1.3))]
    for batched, speech in zip(vad.probabilities_batch(speeches), speeches):
        np.testing.assert_allclose(
            batched, vad.probabilities(speech), rtol=1e-4, atol=1e-5
        )


def test_detect_files_keeps_the_order_of_the_paths(tmp_path):
    vad = _model(tmp_path / "exp")
    vad.threshold = 0.0  # every frame is speech: one segment a file
    paths = []
    for i, seconds in enumerate((1.5, 0.5, 1.0, 0.01)):
        path = tmp_path / f"{i}.wav"
        soundfile.write(path, _speech(seconds, i), FS)
        paths.append(path)
    results = vad.detect_files(paths, batch_size=2)
    assert [len(r) for r in results] == [1, 1, 1, 0]  # 0.01 s has no frame
    ends = [r[0].end for r in results[:3]]
    assert ends[0] > ends[2] > ends[1]
    assert results[:3] == [vad.detect_file(p) for p in paths[:3]]
    with pytest.raises(ValueError):
        vad.detect_files(paths, batch_size=0)


def test_offline_reads_the_largest_look_ahead_and_online_none():
    vad, encoder = _stub_detector()
    assert vad.offline_look_ahead == 6
    vad.detect(_speech(0.5), FS, offline=True)
    vad.detect(_speech(0.5), FS, offline=False)
    vad.stream().accept(_speech(0.5))
    # the stream sets look-ahead 0 when it starts and again on each call
    assert encoder.set == [6, 0, 0, 0]


def test_a_look_ahead_beyond_the_encoder_is_refused():
    vad, _ = _stub_detector()
    with pytest.raises(ValueError, match="exceeds"):
        vad.probabilities(_speech(0.5), FS, look_ahead=7)


def test_read_audio_decodes_only_what_it_is_asked_for(tmp_path):
    from espnet2.bin.diar_inference import audio_seconds

    soundfile.write(tmp_path / "a.wav", _speech(3.0), FS)
    assert audio_seconds(tmp_path / "a.wav") == 3.0
    assert len(read_audio(tmp_path / "a.wav", FS, max_seconds=1.25)) == 20000
    assert len(read_audio(tmp_path / "a.wav", FS)) == 3 * FS


@pytest.mark.parametrize("center, frames", [(False, 0), (True, 400 // 128 + 1)])
def test_a_short_input_has_the_frames_the_stft_gives_it(tmp_path, center, frames):
    """400 samples: no frame of a 512-point STFT, but a centred one pads it."""
    vad = _model(tmp_path, center=center)
    assert vad.min_samples == (257 if center else 512)
    assert len(vad.probabilities(_speech(0.025))) == frames == vad.num_frames(400)


def test_an_encoder_that_subsamples_is_refused(tmp_path, monkeypatch):
    vad = _model(tmp_path)
    call = type(vad.diarize).__call__

    def every_other_frame(self, speech, fs):
        waves, probs = call(self, speech, fs=fs)
        return waves, probs[:, ::2]

    monkeypatch.setattr(type(vad.diarize), "__call__", every_other_frame)
    with pytest.raises(ValueError, match="subsamples"):
        vad.probabilities(_speech(1.0))


def test_a_batch_with_short_recordings_gives_them_no_frames(tmp_path):
    vad = _model(tmp_path)
    speeches = [_speech(0.005), _speech(1.0), _speech(0.01)]
    probs = vad.probabilities_batch(speeches)
    assert [len(p) for p in probs] == [0, vad.num_frames(FS), 0]
    np.testing.assert_allclose(
        probs[1], vad.probabilities(speeches[1]), rtol=1e-4, atol=1e-5
    )
    assert vad.probabilities_batch([_speech(0.001)])[0].shape == (0,)


def test_detect_files_decodes_one_batch_at_a_time(tmp_path, monkeypatch):
    from espnet2.bin import diar_inference as vad_module

    vad = _model(tmp_path / "exp")
    paths = []
    for i, seconds in enumerate((1.5, 0.5, 1.0, 0.75)):
        path = tmp_path / f"{i}.wav"
        soundfile.write(path, _speech(seconds, i), FS)
        paths.append(path)
    events = []
    read = vad_module.read_audio
    batch = vad.probabilities_batch
    monkeypatch.setattr(
        vad_module,
        "read_audio",
        lambda p, fs, **kw: events.append(("read", p)) or read(p, fs, **kw),
    )
    monkeypatch.setattr(
        vad,
        "probabilities_batch",
        lambda s, la=0: events.append(("run", len(s))) or batch(s, la),
    )
    vad.detect_files(paths, batch_size=2)
    # shortest first, two files read before each run
    assert events == [
        ("read", paths[1]),
        ("read", paths[3]),
        ("run", 2),
        ("read", paths[2]),
        ("read", paths[0]),
        ("run", 2),
    ]


def test_integer_pcm_is_scaled_and_one_column_is_squeezed():
    from espnet2.bin.diar_inference import as_mono_float

    np.testing.assert_allclose(
        as_mono_float(np.array([-32768, 0, 16384], dtype=np.int16)), [-1.0, 0.0, 0.5]
    )
    np.testing.assert_allclose(
        as_mono_float(np.array([0, 128, 255], dtype=np.uint8)), [-1.0, 0.0, 127 / 128]
    )
    assert as_mono_float(np.zeros((7, 1))).shape == (7,)
    assert as_mono_float(np.zeros((1, 7))).shape == (7,)
    assert as_mono_float(np.zeros(3, dtype=np.float64)).dtype == np.float32


@pytest.mark.parametrize(
    "bad, message",
    [
        (np.zeros((100, 2)), "one channel"),
        (np.zeros((2, 3, 4)), "one channel"),
        (np.array([0.0, np.nan, 0.0]), "NaN"),
        (np.array([0.0, np.inf]), "NaN"),
    ],
)
def test_bad_audio_is_refused_by_every_entry_point(tmp_path, bad, message):
    vad = _model(tmp_path)
    with pytest.raises(ValueError, match=message):
        vad.probabilities(bad)
    with pytest.raises(ValueError, match=message):
        vad.probabilities_batch([_speech(1.0), bad])
    with pytest.raises(ValueError, match=message):
        vad.stream().accept(bad)


def test_int16_input_gives_what_the_float_input_gives(tmp_path):
    vad = _model(tmp_path)
    speech = np.clip(_speech(1.0) * 4, -1, 1)
    pcm = np.round(speech * 32767).astype(np.int16)
    np.testing.assert_allclose(
        vad.probabilities(pcm),
        vad.probabilities(pcm.astype(np.float32) / 32768),
        atol=1e-6,
    )
    stream = vad.stream()
    for i in range(0, len(pcm), 1000):
        stream.accept(pcm[i : i + 1000])
    np.testing.assert_allclose(
        stream.all_probabilities(), vad.probabilities(pcm), atol=1e-5
    )


def test_a_nonpositive_rate_is_refused(tmp_path):
    vad = _model(tmp_path)
    with pytest.raises(ValueError, match="sample rate"):
        vad.probabilities(_speech(0.5), fs=0)
    with pytest.raises(ValueError, match="sample rate"):
        vad.stream().accept(_speech(0.5), fs=-16000)


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_any_rate_is_resampled_to_the_models(tmp_path, rate):
    """probabilities(x, rate) is the model's rate version of x, frame for frame."""
    from espnet2.bin.diar_inference import resample

    vad = _model(tmp_path)
    x = (np.random.default_rng(rate).normal(size=rate) * 0.1).astype(np.float32)
    at_model_rate = resample(x, rate, FS)
    probs = vad.probabilities(x, rate)
    assert len(probs) == vad.num_frames(len(at_model_rate))
    np.testing.assert_allclose(probs, vad.probabilities(at_model_rate), atol=1e-6)


@pytest.mark.parametrize(
    "rate, subtype, channels",
    [
        (8000, "PCM_16", 1),
        (44100, "PCM_24", 2),
        (48000, "FLOAT", 1),
        (16000, "PCM_16", 3),
    ],
)
def test_files_at_any_rate_format_and_channel_count(tmp_path, rate, subtype, channels):
    """A file's segments are those of its first channel at the model's rate."""
    from espnet2.bin.diar_inference import resample

    vad = _model(tmp_path / "exp")
    rng = np.random.default_rng(rate + channels)
    x = (rng.normal(size=(rate, channels)) * 0.1).clip(-0.9, 0.9)
    ext = "flac" if subtype == "PCM_24" else "wav"
    path = tmp_path / f"a.{ext}"
    soundfile.write(path, x, rate, subtype=subtype)
    first, _ = soundfile.read(path, dtype="float32", always_2d=True)
    expected = vad.probabilities(resample(first[:, 0], rate, FS))
    np.testing.assert_allclose(
        vad.probabilities(read_audio(path, FS)), expected, atol=1e-6
    )
    vad.threshold = 0.0
    assert vad.detect_file(path) == vad.segments(expected)
    assert vad.detect_files([path, path], batch_size=2) == [vad.segments(expected)] * 2


@pytest.mark.parametrize(
    "n", [0, 1, 511, 512, 513, 639, 640, 641, 512 + 128 * 7, 512 + 128 * 7 - 1]
)
def test_frame_counts_at_the_boundaries(tmp_path, n):
    """0 frames below n_fft, then one more frame every hop samples."""
    vad = _model(tmp_path)
    expected = 0 if n < 512 else (n - 512) // 128 + 1
    assert vad.num_frames(n) == expected
    speech = _speech(1.0)[:n]
    assert len(vad.probabilities(speech)) == expected
    assert [len(p) for p in vad.probabilities_batch([speech, _speech(1.0)])][
        0
    ] == expected
    stream = vad.stream()
    stream.accept(speech)
    assert len(stream.all_probabilities()) == expected


def test_empty_inputs_everywhere(tmp_path):
    vad = _model(tmp_path / "exp")
    empty = np.zeros(0, dtype=np.float32)
    assert vad.detect(empty) == []
    assert vad.probabilities_batch([]) == []
    assert vad.probabilities_batch([empty, empty])[1].shape == (0,)
    assert vad.detect_files([]) == []
    path = tmp_path / "empty.wav"
    soundfile.write(path, empty, FS)
    assert vad.detect_file(path) == [] and vad.detect_files([path]) == [[]]
    stream = vad.stream()
    assert stream.accept(empty).shape == (0,)
    assert stream.finish().shape == (0,) and stream.segments() == []


@pytest.mark.parametrize(
    "probs, threshold, median, expected",
    [
        ([1.0] * 5, 0.5, 1, [(0, 5)]),  # all speech
        ([1.0] * 5, 1.0, 1, []),  # not above a threshold of 1
        ([0.0] * 5, 0.0, 1, []),  # not above a threshold of 0
        ([0.9], 0.5, 11, []),  # a median longer than the recording
        ([0.9, 0.9, 0.0, 0.9, 0.9], 0.5, 3, [(0, 5)]),  # a gap filled
        ([0.0, 0.9, 0.0, 0.0, 0.0], 0.5, 3, []),  # a blip removed
    ],
)
def test_the_rule_at_its_edges(probs, threshold, median, expected):
    found = frames_to_segments(np.array(probs), 1.0, threshold, median)
    assert [(s.start, s.end) for s in found] == [
        (float(a), float(b)) for a, b in expected
    ]


def test_segment_times_of_a_long_recording_are_exact():
    """Frame 10 hours in: 128 * frame / 16000 s, no float drift."""
    frames = 10 * 3600 * 125 + 3
    probs = np.zeros(frames, dtype=np.float32)
    probs[-3:] = 1.0
    (seg,) = frames_to_segments(probs, 128 / 16000, 0.5, 1)
    assert seg.start == (frames - 3) * 128 / 16000 and seg.end == frames * 0.008
