from types import SimpleNamespace

import numpy as np
import pytest

from espnet2.sds.vad.frontia_vad import FrontiaVADModel

FS = 16000
HOP = 128
N_FFT = 512


class _Stream:
    """Return a fixed probability for each frame the audio completes."""

    def __init__(self, level):
        self.level = level
        self.samples = 0
        self.emitted = 0

    def accept(self, chunk, fs):
        assert fs == FS
        self.samples += len(chunk)
        available = max(0, (self.samples - N_FFT) // HOP + 1)
        new = available - self.emitted
        self.emitted = available
        return np.asarray(
            [self.level(i) for i in range(available - new, available)],
            dtype=np.float32,
        )


def _detector(level, median=1, threshold=0.5):
    return SimpleNamespace(
        fs=FS,
        n_fft=N_FFT,
        hop=HOP,
        frame_shift=HOP / FS,
        threshold=threshold,
        median=median,
        stream=lambda keep_probabilities=True, look_ahead=0: _Stream(level),
    )


def _speech_between(first, last):
    """Speech in frames first..last-1, silence elsewhere."""
    return lambda i: 0.9 if first <= i < last else 0.1


def _feed(vad, seconds, chunk_ms=20, **kwargs):
    step = FS * chunk_ms // 1000
    audio = np.zeros(int(seconds * FS), dtype=np.int16)
    for i in range(0, len(audio), step):
        out = vad.forward(audio[i : i + step], FS, **kwargs)
        if out is not None:
            return out, i + step
    return None, len(audio)


def test_find_endpoint():
    d = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0])
    assert FrontiaVADModel._find_endpoint(d, 2) == 6
    assert FrontiaVADModel._find_endpoint(d, 4) is None
    assert FrontiaVADModel._find_endpoint(np.zeros(10), 1) is None


def test_decisions_hold_back_the_median_window():
    vad = FrontiaVADModel.from_detector(_detector(lambda i: 0.9, median=5))
    assert len(vad._decisions(np.full(10, 0.9))) == 8


def test_forward_returns_the_turn_after_the_hangover():
    # Speech in frames 50..150 (0.4 s to 1.2 s); the 500 ms hangover is 62
    # frames, so the turn ends once frame 211 has a decision.
    vad = FrontiaVADModel.from_detector(_detector(_speech_between(50, 150)))
    turn, fed = _feed(vad, 3.0, binary=True)
    assert turn is not None
    turn_info = vad.last_turn
    assert turn_info["speech_start_s"] == pytest.approx(50 * HOP / FS)
    assert turn_info["endpoint_s"] == pytest.approx(150 * HOP / FS)
    # 300 ms of padding on each side
    assert len(turn) == int(turn_info["end_s"] * FS) - int(turn_info["start_s"] * FS)
    assert turn_info["end_s"] - turn_info["endpoint_s"] == pytest.approx(0.3)
    # The turn ended within one chunk of the hangover, not at the end of input.
    assert fed < (211 * HOP + N_FFT) + FS * 20 // 1000 + 1


def test_forward_returns_float_audio_at_target_sr():
    vad = FrontiaVADModel.from_detector(
        _detector(_speech_between(50, 150)), target_sr=8000
    )
    turn, _ = _feed(vad, 3.0)
    info = vad.last_turn
    expected = (int(info["end_s"] * FS) - int(info["start_s"] * FS)) // 2
    assert turn.dtype == np.float32
    assert abs(len(turn) - expected) <= 1


def test_forward_drops_a_short_turn():
    # 10 frames is 80 ms, below the 200 ms minimum.
    vad = FrontiaVADModel.from_detector(_detector(_speech_between(50, 60)))
    turn, _ = _feed(vad, 2.0)
    assert turn is None


def test_forward_waits_while_speech_goes_on():
    vad = FrontiaVADModel.from_detector(_detector(lambda i: 0.9))
    turn, _ = _feed(vad, 2.0)
    assert turn is None


def test_pad_end_must_not_exceed_the_endpoint_delay():
    with pytest.raises(ValueError, match="pad_end_ms"):
        FrontiaVADModel.from_detector(
            _detector(lambda i: 0.1), hangover_ms=200, pad_end_ms=300
        )


def test_algorithmic_delay():
    vad = FrontiaVADModel.from_detector(_detector(lambda i: 0.1, median=25))
    # 512 samples of window and 12 frames of median look-ahead: 32 + 96 ms
    assert vad.algorithmic_delay_s == pytest.approx(0.128)
    # 62 frames of hangover: the last one is the 61st after the endpoint
    assert vad.endpoint_delay_s == pytest.approx(0.128 + 61 * HOP / FS)


@pytest.mark.parametrize("median", [1, 25])
def test_endpoint_delay_is_when_the_turn_comes_back(median):
    """Fed one sample at a time, the turn returns endpoint_delay_s after it."""
    vad = FrontiaVADModel.from_detector(
        _detector(_speech_between(50, 150), median=median)
    )
    n = 0
    while vad.forward(np.zeros(1, dtype=np.int16), FS) is None:
        n += 1
    returned_s = (n + 1) / FS
    assert returned_s - vad.last_turn["endpoint_s"] == pytest.approx(
        vad.endpoint_delay_s
    )


def test_end_padding_up_to_the_endpoint_delay_is_all_there():
    """500 ms of end padding is all there although hangover rounds to 496 ms.

    The turn returns 520 ms after the endpoint: 61 frames, plus the window.
    """
    vad = FrontiaVADModel.from_detector(
        _detector(_speech_between(50, 150)), hangover_ms=500, pad_end_ms=500
    )
    _feed(vad, 3.0, binary=True)
    info = vad.last_turn
    assert info["end_s"] - info["endpoint_s"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="endpoint_delay_s"):
        FrontiaVADModel.from_detector(
            _detector(lambda i: 0.1), hangover_ms=500, pad_end_ms=530
        )


def test_tiny_model_end_to_end(tmp_path):
    """Run a randomly initialised causal TCN through StreamingVAD."""
    import torch

    from espnet2.bin.diar_inference import DiarizeSpeech, VoiceActivityDetector
    from espnet2.tasks.diar import DiarizationTask

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
            f"n_fft={N_FFT}",
            "--frontend_conf",
            f"hop_length={HOP}",
            "--frontend_conf",
            "center=false",
            "--normalize",
            "none",
            "--encoder",
            "tcn",
            "--encoder_conf",
            "causal=true",
            "--encoder_conf",
            "num_blocks=2",
            "--encoder_conf",
            "num_repeats=1",
            "--encoder_conf",
            "output_size=8",
        ]
    )
    torch.manual_seed(0)
    detector = VoiceActivityDetector(
        DiarizeSpeech(train_config=tmp_path / "config.yaml", num_spk=1),
        threshold=0.5,
        median=5,
    )
    vad = FrontiaVADModel.from_detector(detector)
    vad.warmup()
    audio = (np.random.default_rng(0).normal(size=10 * FS) * 0.1).astype(np.float32)
    for i in range(0, len(audio), 320):
        vad.forward(audio[i : i + 320], FS)
    # The stream the wrapper runs keeps no probability history and only the
    # samples of the next frame, however long it runs.
    assert vad.frame_vad.stream.cached is not None
    assert vad.frame_vad.stream.probs == []
    assert len(vad.frame_vad.stream.buffer) < N_FFT + 320
    with pytest.raises(ValueError, match="keeps no probabilities"):
        vad.frame_vad.stream.all_probabilities()


def test_forward_copies_the_chunk():
    """Overwriting the caller's buffer after forward() leaves the turn intact."""
    vad = FrontiaVADModel.from_detector(_detector(_speech_between(50, 150)))
    # Chunk n holds the value 1000 + n % 7, written into the same buffer.
    buf = np.empty(320, dtype=np.int16)
    for n in range(3 * FS // 320):
        buf[:] = 1000 + n % 7
        turn = vad.forward(buf, FS, binary=True)
        if turn is not None:
            break
    assert turn is not None
    # Each 320-sample chunk keeps its own value; a kept view would show the
    # last value everywhere.
    assert len(np.unique(turn)) == 7


def test_buffers_stay_bounded_in_silence():
    vad = FrontiaVADModel.from_detector(_detector(lambda i: 0.1, median=25))
    chunk = np.zeros(FS // 10, dtype=np.int16)
    for _ in range(600):  # one minute
        assert vad.forward(chunk, FS) is None
    # pad_start_ms of audio and the median and hold-back frames, not a minute
    assert len(vad.audio) <= 0.3 * FS + FS // 10 + 2 * 25 * HOP
    assert len(vad.probs) <= 2 * 25 + FS // 10 // HOP + 2


def test_two_turns_in_one_chunk_come_back_as_one():
    # speech in frames 50..150 and 250..350, both ended within a 5 s chunk
    def level(i):
        return 0.9 if 50 <= i < 150 or 250 <= i < 350 else 0.1

    vad = FrontiaVADModel.from_detector(_detector(level))
    chunk = (np.arange(5 * FS) % 1000).astype(np.int16)
    turn = vad.forward(chunk, FS, binary=True)
    info = vad.last_turn
    # one turn, from the first one's start to the second one's end
    assert [t["speech_start_s"] for t in info["turns"]] == pytest.approx(
        [50 * HOP / FS, 250 * HOP / FS]
    )
    assert info["speech_start_s"] == pytest.approx(50 * HOP / FS)
    assert info["endpoint_s"] == pytest.approx(350 * HOP / FS)
    a, b = int(round(info["start_s"] * FS)), int(round(info["end_s"] * FS))
    np.testing.assert_array_equal(turn, chunk[a:b])
    # nothing is left for a later call
    assert vad.forward(np.zeros(0, dtype=np.int16), FS) is None


def test_memory_bounded_with_several_turns_in_every_chunk():
    """5 s chunks, each ending two turns: what is held does not grow."""
    from test.memory_growth import assert_memory_bounded

    period = FS // HOP  # a turn every second: 0.5 s speech, 0.5 s silence
    vad = FrontiaVADModel.from_detector(
        _detector(lambda i: 0.9 if i % (2 * period) < period // 2 else 0.1)
    )
    chunk = np.zeros(5 * FS, dtype=np.int16)
    returned = []

    def step():
        if vad.forward(chunk, FS) is not None:
            returned.append(len(vad.last_turn["turns"]))

    assert_memory_bounded(vad, step, steps=240)  # 20 minutes
    assert max(returned) >= 2 and len(returned) >= 239


def test_turn_after_a_short_one_in_the_same_chunk():
    # 80 ms of speech (dropped), then a real turn, in one chunk
    def level(i):
        return 0.9 if 50 <= i < 60 or 150 <= i < 250 else 0.1

    vad = FrontiaVADModel.from_detector(_detector(level))
    turn = vad.forward(np.zeros(4 * FS, dtype=np.int16), FS)
    assert turn is not None
    assert vad.last_turn["speech_start_s"] == pytest.approx(150 * HOP / FS)


def test_int16_then_float_chunks_keep_their_scale():
    vad = FrontiaVADModel.from_detector(
        _detector(_speech_between(50, 150)), target_sr=FS
    )
    # int16 speech at half scale, then float silence
    out = vad.forward(np.full(FS, 16384, dtype=np.int16), FS)
    assert out is None
    out = vad.forward(np.zeros(2 * FS, dtype=np.float32), FS)
    assert out is not None
    assert np.abs(out).max() == pytest.approx(0.5, abs=0.01)


def test_refuses_a_stream_that_keeps_all_audio():
    detector = _detector(lambda i: 0.1)
    detector.stream = lambda keep_probabilities=True, look_ahead=0: SimpleNamespace(
        cached=None, context=None
    )
    with pytest.raises(ValueError, match="bounded audio"):
        FrontiaVADModel.from_detector(detector)


def _tiny_detector(tmp_path, threshold=0.5):
    import torch

    from espnet2.bin.diar_inference import DiarizeSpeech, VoiceActivityDetector
    from espnet2.tasks.diar import DiarizationTask

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
            f"n_fft={N_FFT}",
            "--frontend_conf",
            f"hop_length={HOP}",
            "--frontend_conf",
            "center=false",
            "--normalize",
            "none",
            "--encoder",
            "tcn",
            "--encoder_conf",
            "causal=true",
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
        DiarizeSpeech(train_config=tmp_path / "config.yaml", num_spk=1),
        threshold=threshold,
        median=5,
    )


@pytest.mark.execution_timeout(60)
@pytest.mark.parametrize("stream_keeps_history", [False, True])
def test_idle_memory_bounded(tmp_path, stream_keeps_history):
    """Ten minutes of audio with no speech: what is held must not grow at all.

    With no turn under way the buffers hold a fixed amount, so this finds a
    slow leak that the size of a turn in progress would hide. The case with
    the stream keeping its probability history (the defect CodeRabbit found
    on #7039) shows the check finds it.
    """
    from test.memory_growth import assert_memory_bounded

    # threshold 1.0: no frame is speech
    vad = FrontiaVADModel.from_detector(_tiny_detector(tmp_path, threshold=1.0))
    if stream_keeps_history:
        vad.frame_vad.stream = vad.frame_vad.detector.stream(keep_probabilities=True)
    chunk = (np.random.default_rng(0).normal(size=FS) * 3000).astype(np.int16)

    def run():
        assert_memory_bounded(vad, lambda: vad.forward(chunk, FS), steps=600)

    if stream_keeps_history:
        with pytest.raises(AssertionError, match=r"grown most: frame_vad.stream"):
            run()
    else:
        run()


@pytest.mark.execution_timeout(60)
def test_memory_bounded_over_ten_minutes_of_audio(tmp_path):
    """Ten minutes through a real StreamingVAD in 1 s chunks keep bounded memory.

    The model is random, so its decisions are whatever they are; memory must
    not grow either way.
    """
    from test.memory_growth import assert_memory_bounded

    vad = FrontiaVADModel.from_detector(_tiny_detector(tmp_path))
    rng = np.random.default_rng(0)
    chunk = (rng.normal(size=FS) * 3000).astype(np.int16)
    assert_memory_bounded(vad, lambda: vad.forward(chunk, FS), steps=600)


def test_memory_bounded_over_many_turns():
    """A turn every 2 s for 20 minutes: buffers and queue stay bounded."""
    from test.memory_growth import assert_memory_bounded

    period = 2 * FS // HOP  # frames per 2 s
    vad = FrontiaVADModel.from_detector(
        _detector(lambda i: 0.9 if i % period < period // 2 else 0.1)
    )
    chunk = np.zeros(FS // 2, dtype=np.int16)
    assert_memory_bounded(vad, lambda: vad.forward(chunk, FS), steps=2400)


def test_max_speech_ends_a_turn_that_never_pauses():
    # speech without a pause; turns end every 2 s of speech
    vad = FrontiaVADModel.from_detector(_detector(lambda i: 0.9), max_turn_ms=2000)
    turns = []
    for _ in range(14):  # 7 s in 0.5 s chunks
        out = vad.forward(np.zeros(FS // 2, dtype=np.int16), FS, binary=True)
        if out is not None:
            turns.append(dict(vad.last_turn))
    assert len(turns) == 3
    assert all(t["forced"] for t in turns)
    starts = [t["speech_start_s"] for t in turns]
    assert np.diff(starts) == pytest.approx([2.0, 2.0])


def test_max_speech_below_min_speech_is_refused():
    with pytest.raises(ValueError, match="max_turn_ms"):
        FrontiaVADModel.from_detector(
            _detector(lambda i: 0.1), min_speech_ms=300, max_turn_ms=200
        )


def test_equal_min_and_max_speech_still_return_turns():
    """500 ms both ways rounds to the same 62 frames, so forced turns pass."""
    vad = FrontiaVADModel.from_detector(
        _detector(lambda i: 0.9), min_speech_ms=500, max_turn_ms=500
    )
    turn, _ = _feed(vad, 2.0, binary=True)
    assert turn is not None and vad.last_turn["forced"]


def test_forced_turn_waits_for_its_end_padding():
    """20 ms chunks: the forced turn still carries all 300 ms of end padding."""
    vad = FrontiaVADModel.from_detector(_detector(lambda i: 0.9), max_turn_ms=2000)
    turn, _ = _feed(vad, 4.0, binary=True)
    info = vad.last_turn
    assert turn is not None and info["forced"]
    assert info["end_s"] - info["endpoint_s"] == pytest.approx(0.3)
    assert len(turn) == int(info["end_s"] * FS) - int(info["start_s"] * FS)


def _capture_load(monkeypatch):
    """Make VoiceActivityDetector.load record its arguments and load nothing."""
    from espnet2.bin import diar_inference

    calls = []

    def load(tag, device, threshold, median):
        calls.append(dict(tag=tag, device=device, threshold=threshold, median=median))
        return _detector(lambda i: 0.1, median=median, threshold=threshold)

    monkeypatch.setattr(diar_inference.VoiceActivityDetector, "load", load)
    return calls


def test_defaults_come_from_the_online_vad(monkeypatch):
    from espnet2.bin.diar_inference import VAD_MODELS, VAD_RULES

    calls = _capture_load(monkeypatch)
    FrontiaVADModel()
    threshold, median = VAD_RULES["online"]
    assert calls == [
        dict(
            tag=VAD_MODELS["online"], device="cuda", threshold=threshold, median=median
        )
    ]


def test_arguments_override_the_online_vad(monkeypatch):
    calls = _capture_load(monkeypatch)
    FrontiaVADModel(tag="some/vad", threshold=0.3, median=1, device="cpu")
    assert calls == [dict(tag="some/vad", device="cpu", threshold=0.3, median=1)]


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
@pytest.mark.parametrize("chunk_s", [0.01, 0.1, 0.5])
def test_tiny_model_scores_other_rates_as_their_16k_resampling(tmp_path, rate, chunk_s):
    """Audio at another rate is resampled as one stream before the model.

    The frame scores equal those of the whole recording resampled to 16 kHz
    at once, whatever the chunk length.
    """
    import soxr

    from espnet2.sds.vad.frontia_vad import FrontiaFrameVAD

    detector = _tiny_detector(tmp_path)
    x = (np.random.default_rng(1).normal(size=3 * rate) * 0.1).astype(np.float32)
    frame_vad = FrontiaFrameVAD(detector)
    step = int(chunk_s * rate)
    streamed = np.concatenate(
        [frame_vad.accept(x[i : i + step], rate) for i in range(0, len(x), step)]
    )
    whole = detector.probabilities(soxr.resample(x, rate, FS), FS)
    # The stream resampler (soxr HQ) holds back up to ~120 ms at these rates
    # (measured), so the stream has up to 15 frames fewer.
    assert len(whole) - 15 <= len(streamed) <= len(whole)
    np.testing.assert_allclose(streamed, whole[: len(streamed)], atol=1e-4)


def _dual_mode_detector(tmp_path, threshold=0.5, median=1, seed=0):
    import torch

    from espnet2.bin.diar_inference import DiarizeSpeech, VoiceActivityDetector
    from espnet2.tasks.diar import DiarizationTask

    conf = dict(fs=FS, n_fft=N_FFT, hop_length=HOP, center="false")
    cmd = ["--dry_run", "true", "--output_dir", str(tmp_path), "--num_spk", "1"]
    for k, v in conf.items():
        cmd += ["--frontend_conf", f"{k}={v}"]
    cmd += ["--normalize", "none", "--encoder", "tcn"]
    for kv in ("future_taps=2", "num_blocks=2", "num_repeats=1", "output_size=8"):
        cmd += ["--encoder_conf", kv]
    if not (tmp_path / "config.yaml").exists():
        DiarizationTask.main(cmd=cmd)
    torch.manual_seed(seed)
    return VoiceActivityDetector(
        DiarizeSpeech(train_config=tmp_path / "config.yaml", num_spk=1),
        threshold=threshold,
        median=median,
    )


@pytest.mark.parametrize("look_ahead", [0, 3, 6])
def test_look_ahead_turns_equal_the_whole_recording_rule(tmp_path, look_ahead):
    """Streamed with look-ahead, the frames are those of the whole recording.

    Each turn's probabilities and decisions equal the whole recording's,
    scored once at the same look-ahead through the same rule, and each turn
    returns endpoint_delay_s after its endpoint, which includes the
    look-ahead.
    """
    rng = np.random.default_rng(3)
    # loud bursts in silence; the threshold sits between the two levels the
    # random model gives them, so that the bursts make turns
    audio = np.zeros(12 * FS, dtype=np.float32)
    burst = np.zeros(len(audio), dtype=bool)
    for a in (1.0, 4.0, 8.0):
        n = int(1.5 * FS)
        audio[int(a * FS) : int(a * FS) + n] = rng.normal(size=n) * 0.5
        burst[int(a * FS) : int(a * FS) + n] = True
    # the first of these random models that scores the bursts higher
    for seed in range(20):
        detector = _dual_mode_detector(tmp_path, median=3, seed=seed)
        whole = detector.probabilities(audio, FS, look_ahead)
        frame_burst = burst[np.arange(len(whole)) * HOP + N_FFT // 2]
        hi, lo = np.median(whole[frame_burst]), np.median(whole[~frame_burst])
        if hi - lo > 1e-3:
            break
    assert hi - lo > 1e-3
    vad = FrontiaVADModel.from_detector(
        detector,
        look_ahead=look_ahead,
        threshold=float((hi + lo) / 2),
        min_speech_ms=8.0,
        hangover_ms=300.0,
    )
    assert vad.frame_vad.window_s == pytest.approx((N_FFT + look_ahead * HOP) / FS)
    decisions = vad._decisions(whole)
    turns = []
    for i in range(0, len(audio), 160):
        if vad.forward(audio[i : i + 160], FS) is not None:
            turns.append((dict(vad.last_turn), (i + 160) / FS))
    assert len(turns) == 3
    for t, returned_s in turns:
        k = int(round(t["probs_start_s"] / (HOP / FS)))
        np.testing.assert_allclose(
            t["probs"], whole[k : k + len(t["probs"])], atol=1e-5
        )
        np.testing.assert_array_equal(
            t["decisions"], decisions[k : k + len(t["decisions"])]
        )
        late = returned_s - t["endpoint_s"] - vad.endpoint_delay_s
        assert -1e-9 <= late < 0.01 + 1e-9  # within the 10 ms chunk
