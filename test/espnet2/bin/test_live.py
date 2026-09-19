"""Streaming transcription: the window loop, the sources, and the errors."""

import sys
import types

import numpy as np
import pytest
import soundfile as sf

from espnet2.bin import live


def blocks(*sizes):
    return (np.full(size, i + 1, dtype=np.float32) for i, size in enumerate(sizes))


def test_blocks_are_regrouped_into_windows():
    # the microphone's block size has nothing to do with the model's window
    got = list(live.windows(blocks(3, 3, 3), window=4))
    assert [len(w) for w in got] == [4, 4, 1]


def test_the_last_partial_window_is_still_decoded():
    # otherwise the end of a recording disappears without a word
    assert [len(w) for w in live.windows(blocks(5), window=4)] == [4, 1]
    assert [len(w) for w in live.windows(blocks(5), window=4, flush_partial=False)] == [
        4
    ]


def test_a_window_holds_what_the_blocks_held_in_order():
    got = list(live.windows(blocks(2, 2), window=4))
    assert got[0].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_a_file_is_read_in_blocks_and_mixed_to_one_channel(tmp_path):
    path = tmp_path / "stereo.wav"
    stereo = np.stack([np.full(16000, 1.0), np.full(16000, -1.0)], axis=1).astype(
        np.float32
    )
    sf.write(path, stereo, 16000)
    got = np.concatenate(list(live.from_file(str(path), block=4000)))
    assert len(got) == 16000
    # the two channels cancel, so they were mixed; 1.0 does not survive
    # a 16-bit wav exactly, hence the tolerance
    assert np.allclose(got, 0.0, atol=1e-4)


def test_a_file_at_another_rate_is_resampled(tmp_path):
    path = tmp_path / "eight.wav"
    sf.write(path, np.zeros(8000, dtype=np.float32), 8000)
    got = np.concatenate(list(live.from_file(str(path))))
    assert 15000 < len(got) <= 16000  # one second, at the model's rate


def test_transcribe_prints_each_window_as_it_fills():
    said = []
    source = blocks(16000 * 20, 16000 * 20)
    code = live.transcribe(
        lambda chunk: f"{len(chunk)} samples", source, on_text=said.append
    )
    assert code == 0
    assert said == ["320000 samples", "320000 samples"]


def test_an_empty_transcript_is_not_printed():
    said = []
    live.transcribe(lambda chunk: "", blocks(16000 * 20), on_text=said.append)
    assert said == []


def test_stopping_a_recording_is_not_an_error():
    def interrupted():
        raise KeyboardInterrupt
        yield  # pragma: no cover - makes this a generator

    assert live.transcribe(lambda chunk: "x", interrupted()) == 130


def test_recording_without_sounddevice_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(live.LiveError) as raised:
        next(live.from_microphone())
    assert "pip install sounddevice" in str(raised.value)


def test_the_microphone_stream_yields_what_the_callback_was_given(monkeypatch, capsys):
    captured = {}

    class Stream:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            # the device hands blocks to the callback; two are enough, and
            # the second arrives with a status, which is how PortAudio
            # reports that blocks were dropped
            captured["callback"](np.full((4, 1), 1.0), 4, None, None)
            captured["callback"](np.full((4, 1), 2.0), 4, None, "input overflow")
            return self

        def __exit__(self, *exc):
            return False

    def make_stream(**kwargs):
        kwargs["callback"] = kwargs.get("callback")
        stream = Stream(**kwargs)
        captured["callback"] = kwargs["callback"]
        return stream

    fake = types.ModuleType("sounddevice")
    fake.InputStream = make_stream
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    source = live.from_microphone(sample_rate=16000, block=4)
    assert next(source).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert next(source).tolist() == [2.0, 2.0, 2.0, 2.0]
    assert captured["samplerate"] == 16000 and captured["channels"] == 1
    assert "input overflow" in capsys.readouterr().err


def test_a_file_is_never_read_whole(tmp_path, monkeypatch):
    # the point of --stream is that memory does not follow the file's
    # length, so reading it in one call is the one thing this must not do
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(16000 * 3, dtype=np.float32), 16000)

    def fail(*args, **kwargs):  # pragma: no cover - the point is it is not called
        raise AssertionError("the whole file was read at once")

    monkeypatch.setattr(sf, "read", fail)
    sizes = [len(b) for b in live.from_file(str(path), block=4000)]
    assert sizes == [4000] * 12
    assert max(sizes) == 4000  # no block is the size of the file


def test_resampling_across_blocks_matches_resampling_the_whole_file(tmp_path):
    # a resampler restarted at every block leaves a discontinuity at each
    # boundary; this is the check that the filter is carried across them
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(0)
    speech = rng.standard_normal(8000 * 2).astype(np.float32) * 0.1
    path = tmp_path / "eight.wav"
    sf.write(path, speech, 8000)

    streamed = np.concatenate(list(live.from_file(str(path), block=1000)))
    whole = librosa.resample(speech, orig_sr=8000, target_sr=16000)

    n = min(len(streamed), len(whole))
    assert abs(len(streamed) - len(whole)) < 100  # same signal, same length
    assert np.corrcoef(streamed[:n], whole[:n])[0, 1] > 0.99


def test_the_block_queue_does_not_grow_past_its_ceiling():
    said = []
    blocks = live.BoundedBlocks(
        maxsize=2, secs_per_block=0.25, warn_every_secs=10.0, report=said.append
    )
    for i in range(5):
        blocks.put(np.full(4, float(i), dtype=np.float32))

    # three were dropped, and what is left is the newest audio: a live
    # transcript that is minutes behind is worse than one with a gap
    assert blocks.dropped == 3
    assert [blocks.get()[0], blocks.get()[0]] == [3.0, 4.0]
    assert blocks.dropped_secs() == pytest.approx(0.75)


def test_dropping_audio_is_reported_at_once_and_then_sparingly():
    said = []
    blocks = live.BoundedBlocks(
        maxsize=1, secs_per_block=1.0, warn_every_secs=4.0, report=said.append
    )
    for _ in range(9):
        blocks.put(np.zeros(4, dtype=np.float32))

    # the first drop is said immediately, then one line per 4 s dropped:
    # silence about lost speech is not an option, a line per block is noise
    assert len(said) == 3
    assert "slower than the audio arrives" in said[0]
    assert "8 s of audio" in said[-1]
