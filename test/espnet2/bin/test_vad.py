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


# --- input: dtypes, channels, bad values ------------------------------------


def test_integer_pcm_is_scaled_and_one_column_is_squeezed():
    from espnet2.bin.vad import as_mono_float

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


# --- sample rates -----------------------------------------------------------


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_any_rate_is_resampled_to_the_models(tmp_path, rate):
    """probabilities(x, rate) is the model's rate version of x, frame for frame."""
    from espnet2.bin.vad import resample

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
    from espnet2.bin.vad import resample

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


@pytest.mark.parametrize("rate", [8000, 44100])
def test_a_stream_at_another_rate_equals_the_whole_resampled_once(tmp_path, rate):
    """Up- and non-integer resampling, chunk by chunk, then finish()."""
    import soxr

    vad = _model(tmp_path)
    x = (np.random.default_rng(rate).normal(size=2 * rate) * 0.1).astype(np.float32)
    whole = vad.probabilities(soxr.resample(x, rate, FS), FS)
    stream = vad.stream()
    rng = np.random.default_rng(1)
    i = 0
    while i < len(x):
        n = int(rng.integers(1, rate // 10))
        stream.accept(x[i : i + n], rate)
        i += n
    stream.finish()
    np.testing.assert_allclose(stream.all_probabilities(), whole, atol=1e-4)


# --- lengths at and around a frame -----------------------------------------


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


def test_a_stream_fed_one_sample_at_a_time(tmp_path):
    vad = _model(tmp_path)
    speech = _speech(0.2)
    stream = vad.stream()
    for x in speech:
        stream.accept(np.array([x], dtype=np.float32))
    np.testing.assert_allclose(
        stream.all_probabilities(), vad.probabilities(speech), atol=1e-5
    )


def test_a_stream_fed_once(tmp_path):
    vad = _model(tmp_path)
    speech = _speech(1.5)
    stream = vad.stream()
    np.testing.assert_allclose(
        stream.accept(speech), vad.probabilities(speech), atol=1e-5
    )


def test_two_streams_do_not_share_state(tmp_path):
    vad = _model(tmp_path)
    a, b = _speech(1.0, 1), _speech(1.0, 2)
    sa, sb = vad.stream(), vad.stream()
    for i in range(0, FS, 800):
        sa.accept(a[i : i + 800])
        sb.accept(b[i : i + 800])
    np.testing.assert_allclose(sa.all_probabilities(), vad.probabilities(a), atol=1e-5)
    np.testing.assert_allclose(sb.all_probabilities(), vad.probabilities(b), atol=1e-5)


# --- the decision rule at its edges ----------------------------------------


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


# --- a dual-mode model: online and offline ----------------------------------


def _dual_mode_model(tmp_path):
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
            "center=false",
            "--normalize",
            "none",
            "--encoder",
            "tcn",
            "--encoder_conf",
            "future_taps=2",
            "--encoder_conf",
            "num_blocks=2",
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


def test_a_dual_mode_model_reads_ahead_offline_and_streams_online(tmp_path):
    vad = _dual_mode_model(tmp_path)
    assert vad.offline_look_ahead == 6
    speech = _speech(2.0)
    online = vad.probabilities(speech, FS, 0)
    offline = vad.probabilities(speech, FS, vad.offline_look_ahead)
    assert len(online) == len(offline) and np.abs(online - offline).max() > 1e-4
    stream = vad.stream()
    assert stream.cached is not None  # forward_streaming
    for i in range(0, len(speech), 777):
        stream.accept(speech[i : i + 777])
    np.testing.assert_allclose(stream.all_probabilities(), online, atol=1e-5)
    # streaming leaves offline detection reading ahead
    np.testing.assert_allclose(
        vad.probabilities(speech, FS, vad.offline_look_ahead), offline, atol=1e-6
    )
    batched = vad.probabilities_batch([speech, speech[:20000]], vad.offline_look_ahead)
    np.testing.assert_allclose(batched[0], offline, atol=1e-5)
    np.testing.assert_allclose(
        batched[1],
        vad.probabilities(speech[:20000], FS, vad.offline_look_ahead),
        atol=1e-5,
    )
