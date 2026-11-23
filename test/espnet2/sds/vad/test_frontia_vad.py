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
        stream=lambda keep_probabilities=True: _Stream(level),
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
    assert FrontiaVADModel.find_endpoint(d, 2) == 6
    assert FrontiaVADModel.find_endpoint(d, 4) is None
    assert FrontiaVADModel.find_endpoint(np.zeros(10), 1) is None


def test_decisions_hold_back_the_median_window():
    vad = FrontiaVADModel.from_detector(_detector(lambda i: 0.9, median=5))
    assert len(vad.decisions(np.full(10, 0.9))) == 8


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


def test_pad_end_must_not_exceed_the_hangover():
    with pytest.raises(ValueError, match="pad_end_ms"):
        FrontiaVADModel.from_detector(
            _detector(lambda i: 0.1), hangover_ms=200, pad_end_ms=300
        )


def test_algorithmic_delay():
    vad = FrontiaVADModel.from_detector(_detector(lambda i: 0.1, median=25))
    # 512 samples of window and 12 frames of median look-ahead: 32 + 96 ms
    assert vad.algorithmic_delay_s == pytest.approx(0.128)
    assert vad.endpoint_delay_s == pytest.approx(0.128 + 62 * HOP / FS)


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
    assert vad.stream.cached is not None
    assert vad.stream.probs == []
    assert len(vad.stream.buffer) < N_FFT + 320
    with pytest.raises(ValueError, match="keeps no probabilities"):
        vad.stream.all_probabilities()


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


def test_two_turns_in_one_chunk():
    # speech in frames 50..150 and 250..350, both ended within a 5 s chunk
    level = lambda i: 0.9 if 50 <= i < 150 or 250 <= i < 350 else 0.1  # noqa: E731
    vad = FrontiaVADModel.from_detector(_detector(level))
    first = vad.forward(np.zeros(5 * FS, dtype=np.int16), FS, binary=True)
    assert first is not None
    assert vad.last_turn["speech_start_s"] == pytest.approx(50 * HOP / FS)
    second = vad.forward(np.zeros(0, dtype=np.int16), FS, binary=True)
    assert second is not None
    assert vad.last_turn["speech_start_s"] == pytest.approx(250 * HOP / FS)
    assert vad.last_turn["endpoint_s"] == pytest.approx(350 * HOP / FS)
    assert vad.forward(np.zeros(0, dtype=np.int16), FS) is None


def test_turn_after_a_short_one_in_the_same_chunk():
    # 80 ms of speech (dropped), then a real turn, in one chunk
    level = lambda i: 0.9 if 50 <= i < 60 or 150 <= i < 250 else 0.1  # noqa: E731
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
    detector.stream = lambda keep_probabilities=True: SimpleNamespace(
        cached=None, context=None
    )
    with pytest.raises(ValueError, match="bounded audio"):
        FrontiaVADModel.from_detector(detector)
