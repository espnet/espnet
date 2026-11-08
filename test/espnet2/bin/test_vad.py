"""espnet2.bin.vad: the decision rule, streaming, batching and look-ahead.

The model tests build a small randomly initialised VAD the way the recipe
builds the frontia_vad_v1 models (diar1, num_spk 1, causal TCN, STFT without
centring), so nothing is downloaded; what is checked is that the three ways of
running it agree, not what it detects.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
import torch

from espnet2.bin.diar_inference import DiarizeSpeech
from espnet2.bin.vad import (
    MEDIAN,
    THRESHOLD,
    Segment,
    StreamingVAD,
    VoiceActivityDetector,
    format_segments,
    frames_to_segments,
    read_audio,
)
from espnet2.tasks.diar import DiarizationTask

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


# --- the decision rule ------------------------------------------------------


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
    from espnet2.bin.vad import MODELS, RULES

    assert set(MODELS) == set(RULES) == {"online", "offline"}
    for threshold, median in RULES.values():
        assert 0 < threshold < 1 and median >= 1 and median % 2 == 1
    assert (THRESHOLD, MEDIAN) == RULES["offline"]
    assert all(tag.startswith("espnet/") for tag in MODELS.values())


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_for_mode_loads_the_model_of_the_mode_with_its_rule(monkeypatch, mode):
    from espnet2.bin import vad as vad_module

    loaded = {}

    def load(tag, device="cpu", threshold=None, median=None):
        loaded.update(tag=tag, threshold=threshold, median=median)
        return "detector"

    monkeypatch.setattr(VoiceActivityDetector, "load", staticmethod(load))
    monkeypatch.setitem(vad_module.RULES, mode, (0.35, 7))
    assert VoiceActivityDetector.for_mode(mode) == "detector"
    assert loaded == {"tag": vad_module.MODELS[mode], "threshold": 0.35, "median": 7}
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


# --- the model, three ways --------------------------------------------------


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


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stream_equals_the_whole_recording(tmp_path, seed):
    """Chunks of any size give each frame what the whole recording gives it."""
    vad = _model(tmp_path)
    speech = _speech(4.0, seed)
    whole = vad.probabilities(speech)
    stream = vad.stream()
    rng = np.random.default_rng(seed)
    got, i = [], 0
    while i < len(speech):
        n = int(rng.integers(1, 3000))
        got.append(stream.accept(speech[i : i + n]))
        i += n
    streamed = np.concatenate(got)
    np.testing.assert_allclose(streamed, whole, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(stream.all_probabilities(), whole, rtol=1e-4, atol=1e-5)
    assert len(stream.buffer) < vad.n_fft
    assert stream.segments() == vad.segments(whole)


def test_stream_at_another_rate_is_resampled_as_one_signal(tmp_path):
    import soxr

    vad = _model(tmp_path)
    rng = np.random.default_rng(3)
    speech48 = (rng.normal(size=3 * 48000) * 0.1).astype(np.float32)
    whole = vad.probabilities(soxr.resample(speech48, 48000, FS), FS)
    stream = vad.stream()
    for i in range(0, len(speech48), 4800):
        stream.accept(speech48[i : i + 4800], 48000)
    streamed = stream.all_probabilities()
    # the stream resampler holds back a few samples at the end
    assert len(whole) - 5 <= len(streamed) <= len(whole)
    np.testing.assert_allclose(streamed, whole[: len(streamed)], rtol=1e-4, atol=1e-4)


def test_stream_refuses_a_centred_stft(tmp_path):
    vad = _model(tmp_path, center=True)
    with pytest.raises(ValueError, match="center"):
        vad.stream()


def test_a_causal_tcn_streams_from_a_cache(tmp_path):
    """Each call runs only the new frames; the buffer is under one frame."""
    vad = _model(tmp_path)
    stream = vad.stream()
    assert stream.cached is not None
    speech = _speech(3.0, 4)
    for i in range(0, len(speech), 777):
        stream.accept(speech[i : i + 777])
        # the samples from the next frame's start on, and no more
        assert len(stream.buffer) < vad.n_fft
    np.testing.assert_allclose(
        stream.all_probabilities(), vad.probabilities(speech), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("receptive_field", ["known", None])
def test_the_recomputing_stream_equals_the_whole_recording(tmp_path, receptive_field):
    """The fallback for encoders that cannot be cached."""
    vad = _model(tmp_path)
    if receptive_field is None:
        vad.receptive_field = None
    stream = vad.stream()
    stream.cached = None
    speech = _speech(2.0)
    rng = np.random.default_rng(5)
    i = 0
    while i < len(speech):
        n = int(rng.integers(1, 2000))
        stream.accept(speech[i : i + n])
        i += n
    np.testing.assert_allclose(
        stream.all_probabilities(), vad.probabilities(speech), rtol=1e-4, atol=1e-5
    )
    if receptive_field is None:
        assert stream.offset == 0  # the whole history is kept
    else:
        assert stream.offset > 0


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


# --- look-ahead -------------------------------------------------------------


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


def test_stream_class_is_exported():
    vad, _ = _stub_detector()
    assert isinstance(vad.stream(), StreamingVAD)


# --- reading, short inputs and frame rate ---------------------------------


def test_read_audio_decodes_only_what_it_is_asked_for(tmp_path):
    from espnet2.bin.vad import audio_seconds

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
    from espnet2.bin import vad as vad_module

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


# --- streaming: what it refuses, rates and the end of a stream ------------


def test_streaming_refuses_an_utterance_normalisation(tmp_path):
    from espnet2.layers.utterance_mvn import UtteranceMVN

    vad = _model(tmp_path)
    vad.diarize.diar_model.normalize = UtteranceMVN()
    with pytest.raises(ValueError, match="normalisation"):
        vad.stream()


def test_streaming_refuses_an_encoder_that_reads_ahead(tmp_path):
    """An encoder that is not causal and cannot be set to look-ahead 0."""
    vad = _model(tmp_path)
    vad.diarize.diar_model.encoder = torch.nn.Identity()
    with pytest.raises(ValueError, match="causal encoder"):
        vad.stream()


def test_finish_runs_the_audio_the_resampler_holds(tmp_path):
    import soxr

    vad = _model(tmp_path)
    speech48 = (np.random.default_rng(7).normal(size=2 * 48000) * 0.1).astype(
        np.float32
    )
    whole = vad.probabilities(soxr.resample(speech48, 48000, FS), FS)
    stream = vad.stream()
    for i in range(0, len(speech48), 4800):
        stream.accept(speech48[i : i + 4800], 48000)
    held = len(whole) - len(stream.all_probabilities())
    stream.finish()
    assert held > 0 and len(stream.all_probabilities()) == len(whole)
    np.testing.assert_allclose(stream.all_probabilities(), whole, atol=1e-4)
    assert stream.finish().shape == (0,)  # nothing left


def test_a_new_input_rate_flushes_the_old_one_first(tmp_path):
    import soxr

    vad = _model(tmp_path)
    rng = np.random.default_rng(8)
    first = (rng.normal(size=48000) * 0.1).astype(np.float32)  # 1 s at 48 kHz
    second = _speech(1.0, 9)
    whole = vad.probabilities(
        np.concatenate([soxr.resample(first, 48000, FS), second]), FS
    )
    stream = vad.stream()
    got = [stream.accept(first[i : i + 4800], 48000) for i in range(0, 48000, 4800)]
    got += [stream.accept(second[i : i + 1600], FS) for i in range(0, FS, 1600)]
    np.testing.assert_allclose(np.concatenate(got), whole, atol=1e-4)


def test_a_float64_model_streams(tmp_path):
    vad = _model(tmp_path)
    speech = _speech(1.0)
    expected = vad.probabilities(speech)
    vad.diarize.diar_model.double()
    stream = vad.stream()
    for i in range(0, len(speech), 3000):
        stream.accept(speech[i : i + 3000])
    np.testing.assert_allclose(stream.all_probabilities(), expected, atol=1e-5)


# --- a packed model -------------------------------------------------------


def test_a_packed_model_loads_and_streams_like_the_whole_recording(tmp_path):
    """Pack a model as diar.sh stage 8 does, load the zip, stream it."""
    from espnet2.bin.vad import format_segments
    from espnet2.main_funcs.pack_funcs import pack

    vad = _model(tmp_path / "exp")
    model_file = tmp_path / "exp" / "valid.acc.best.pth"
    torch.save(vad.diarize.diar_model.state_dict(), model_file)
    packed = tmp_path / "packed.zip"
    pack(
        files=dict(model_file=model_file),
        yaml_files=dict(train_config=tmp_path / "exp" / "config.yaml"),
        outpath=packed,
    )
    loaded = VoiceActivityDetector.load(str(packed), threshold=0.4, median=3)
    assert (loaded.center, loaded.threshold, loaded.median) == (False, 0.4, 3)
    speech = _speech(3.0, 11)
    whole = loaded.probabilities(speech)
    np.testing.assert_allclose(whole, vad.probabilities(speech), atol=1e-6)
    stream = loaded.stream()
    assert stream.cached is not None
    for i in range(0, len(speech), 1234):
        stream.accept(speech[i : i + 1234])
    np.testing.assert_allclose(stream.all_probabilities(), whole, atol=1e-5)
    assert format_segments(stream.segments()) == format_segments(loaded.segments(whole))


def test_get_parser_is_the_detect_command():
    from espnet2.bin.vad import get_parser

    args = get_parser().parse_args(["a.wav", "--online"])
    assert args.audio == ["a.wav"] and args.online
