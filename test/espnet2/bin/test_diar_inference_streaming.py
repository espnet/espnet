"""diar_inference_streaming.StreamingVAD: online detection, chunk by chunk.

The models are small and randomly initialised (diar1, num_spk 1, TCN, STFT
without centring); each test checks that a stream gives what the whole
recording gives the same model, or that it refuses what it cannot stream.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from espnet2.bin.diar_inference import DiarizeSpeech, VoiceActivityDetector
from espnet2.bin.diar_inference_streaming import StreamingVAD
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


def test_stream_class_is_exported():
    vad, _ = _stub_detector()
    assert isinstance(vad.stream(), StreamingVAD)


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


def test_a_packed_model_loads_and_streams_like_the_whole_recording(tmp_path):
    """Pack a model as diar.sh stage 8 does, load the zip, stream it."""
    from espnet2.bin.diar_inference import format_segments
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
    from espnet2.bin.diar_inference_streaming import get_parser

    args = get_parser().parse_args(["a.wav", "--online"])
    assert args.audio == ["a.wav"] and args.online


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
