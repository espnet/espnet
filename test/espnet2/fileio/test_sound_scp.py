from pathlib import Path

import numpy as np
import pytest
import soundfile

from espnet2.fileio.sound_scp import SoundScpReader, SoundScpWriter


def test_SoundScpReader(tmp_path: Path):
    audio_path1 = tmp_path / "a1.wav"
    audio1 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio_path2 = tmp_path / "a2.wav"
    audio2 = np.random.randint(-100, 100, 16, dtype=np.int16)

    soundfile.write(audio_path1, audio1, 16)
    soundfile.write(audio_path2, audio2, 16)

    p = tmp_path / "dummy.scp"
    with p.open("w") as f:
        f.write(f"abc {audio_path1}\n")
        f.write(f"def {audio_path2}\n")

    desired = {"abc": (16, audio1), "def": (16, audio2)}
    target = SoundScpReader(p, dtype=np.int16)

    for k in desired:
        rate1, t = target[k]
        rate2, d = desired[k]
        assert rate1 == rate2
        np.testing.assert_array_equal(t, d)

    assert len(target) == len(desired)
    assert "abc" in target
    assert "def" in target
    assert tuple(target.keys()) == tuple(desired)
    assert tuple(target) == tuple(desired)
    assert target.get_path("abc") == str(audio_path1)
    assert target.get_path("def") == str(audio_path2)


def test_SoundScpReader_multi(tmp_path: Path):
    audio_path1 = tmp_path / "a1.wav"
    audio1 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio_path2 = tmp_path / "a2.wav"
    audio2 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio_path3 = tmp_path / "a3.wav"
    audio3 = np.random.randint(-100, 100, 16, dtype=np.int16)

    soundfile.write(audio_path1, audio1, 16)
    soundfile.write(audio_path2, audio2, 16)
    soundfile.write(audio_path3, audio3, 16)

    p = tmp_path / "dummy.scp"
    with p.open("w") as f:
        f.write(f"abc {audio_path1}\n")
        f.write(f"def {audio_path2} {audio_path3}\n")

    desired = {
        "abc": (16, audio1),
        "def": (16, np.concatenate([audio2[:, None], audio3[:, None]], axis=1)),
    }
    target = SoundScpReader(p, multi_columns=True, dtype=np.int16)

    for k in desired:
        rate1, t = target[k]
        rate2, d = desired[k]
        assert rate1 == rate2
        np.testing.assert_array_equal(t, d)

    assert len(target) == len(desired)
    assert "abc" in target
    assert "def" in target
    assert tuple(target.keys()) == tuple(desired)
    assert tuple(target) == tuple(desired)
    assert target.get_path("abc") == [str(audio_path1)]
    assert target.get_path("def") == [str(audio_path2), str(audio_path3)]


def test_SoundScpReader_multi_error(tmp_path: Path):
    audio_path1 = tmp_path / "a1.wav"
    audio1 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio_path2 = tmp_path / "a2.wav"
    audio2 = np.random.randint(-100, 100, 32, dtype=np.int16)
    audio_path3 = tmp_path / "a3.wav"
    audio_path4 = tmp_path / "a4.wav"
    audio3 = np.random.randint(-100, 100, 16, dtype=np.int16)

    soundfile.write(audio_path1, audio1, 16)
    soundfile.write(audio_path2, audio2, 16)
    soundfile.write(audio_path3, audio3, 16)
    soundfile.write(audio_path4, audio3, 32)

    p = tmp_path / "dummy.scp"
    with p.open("w") as f:
        f.write(f"abc {audio_path1} {audio_path2}\n")
    p2 = tmp_path / "dummy2.scp"
    with p2.open("w") as f:
        f.write(f"abc {audio_path3} {audio_path4}\n")

    target = SoundScpReader(p, multi_columns=True, dtype=np.int16)
    target2 = SoundScpReader(p2, multi_columns=True, dtype=np.int16)

    with pytest.raises(RuntimeError):
        for _ in target.items():
            pass
    with pytest.raises(RuntimeError):
        for _ in target2.items():
            pass


def test_SoundScpWriter(tmp_path: Path):
    audio1 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio2 = np.random.randint(-100, 100, 16, dtype=np.int16)
    with SoundScpWriter(tmp_path, tmp_path / "wav.scp") as writer:
        writer["abc"] = 16, audio1
        writer["def"] = audio2, 16
        # Unsupported dimension
        with pytest.raises(RuntimeError):
            y = np.random.randint(-100, 100, [16, 1, 1])
            writer["ghi"] = 16, y
        with pytest.raises(ValueError):
            writer["ghi"] = [None, None, None]
        with pytest.raises(TypeError):
            writer["ghi"] = None, None
    target = SoundScpReader(tmp_path / "wav.scp", dtype=np.int16)
    desired = {"abc": (16, audio1), "def": (16, audio2)}

    for k in desired:
        rate1, t = target[k]
        rate2, d = desired[k]
        assert rate1 == rate2
        np.testing.assert_array_equal(t, d)

    assert writer.get_path("abc") == str(tmp_path / "abc.wav")
    assert writer.get_path("def") == str(tmp_path / "def.wav")


def test_SoundScpWriter_multi_columns(tmp_path: Path):
    audio1 = np.random.randint(-100, 100, [16, 2], dtype=np.int16)
    audio2 = np.random.randint(-100, 100, [16, 3], dtype=np.int16)
    with SoundScpWriter(tmp_path, tmp_path / "wav.scp", multi_columns=True) as writer:
        writer["abc"] = 16, audio1
        writer["def"] = 16, audio2

    target = SoundScpReader(tmp_path / "wav.scp", multi_columns=True, dtype=np.int16)
    desired = {"abc": (16, audio1), "def": (16, audio2)}

    for k in desired:
        rate1, t = target[k]
        rate2, d = desired[k]
        assert rate1 == rate2
        np.testing.assert_array_equal(t, d)

    assert writer.get_path("abc") == [
        str(tmp_path / "abc-CH0.wav"),
        str(tmp_path / "abc-CH1.wav"),
    ]
    assert writer.get_path("def") == [
        str(tmp_path / "def-CH0.wav"),
        str(tmp_path / "def-CH1.wav"),
        str(tmp_path / "def-CH2.wav"),
    ]
