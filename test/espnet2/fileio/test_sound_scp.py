from pathlib import Path

import h5py
import numpy as np
import pytest
import soundfile

from espnet2.fileio.sound_scp import SoundScpReader
from espnet2.fileio.sound_scp import SoundScpWriter


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
    target = SoundScpReader(p, normalize=False, dtype=np.int16)

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


def test_SoundScpWriter(tmp_path: Path):
    audio1 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio2 = np.random.randint(-100, 100, 16, dtype=np.int16)
    with SoundScpWriter(tmp_path, tmp_path / "wav.scp", dtype=np.int16) as writer:
        writer["abc"] = 16, audio1
        writer["def"] = 16, audio2
        # Unsupported dimension
        with pytest.raises(RuntimeError):
            y = np.random.randint(-100, 100, [16, 1, 1], dtype=np.int16)
            writer["ghi"] = 16, y
    target = SoundScpReader(tmp_path / "wav.scp", normalize=False, dtype=np.int16)
    desired = {"abc": (16, audio1), "def": (16, audio2)}

    for k in desired:
        rate1, t = target[k]
        rate2, d = desired[k]
        assert rate1 == rate2
        np.testing.assert_array_equal(t, d)

    assert writer.get_path("abc") == str(tmp_path / "abc.wav")
    assert writer.get_path("def") == str(tmp_path / "def.wav")


def test_SoundScpReader_normalize(tmp_path: Path):
    audio_path1 = tmp_path / "a1.wav"
    audio1 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio_path2 = tmp_path / "a2.wav"
    audio2 = np.random.randint(-100, 100, 16, dtype=np.int16)

    audio1 = audio1.astype(np.float64) / (np.iinfo(np.int16).max + 1)
    audio2 = audio2.astype(np.float64) / (np.iinfo(np.int16).max + 1)

    soundfile.write(audio_path1, audio1, 16)
    soundfile.write(audio_path2, audio2, 16)

    p = tmp_path / "dummy.scp"
    with p.open("w") as f:
        f.write(f"abc {audio_path1}\n")
        f.write(f"def {audio_path2}\n")

    desired = {"abc": (16, audio1), "def": (16, audio2)}
    target = SoundScpReader(p, normalize=True)

    for k in desired:
        rate1, t = target[k]
        rate2, d = desired[k]
        assert rate1 == rate2
        np.testing.assert_array_equal(t, d)


def test_hdf5_SoundScpReader(tmp_path: Path):
    audio_path1 = tmp_path / "a1.wav"
    audio1 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio_path2 = tmp_path / "a2.wav"
    audio2 = np.random.randint(-100, 100, 16, dtype=np.int16)

    soundfile.write(audio_path1, audio1, 16)
    soundfile.write(audio_path2, audio2, 16)

    p = h5py.File(tmp_path / "dummy.h5")
    p["abc"] = str(audio_path1)
    p["def"] = str(audio_path2)

    desired = {"abc": (16, audio1), "def": (16, audio2)}
    target = SoundScpReader(p, normalize=False, dtype=np.int16)

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
