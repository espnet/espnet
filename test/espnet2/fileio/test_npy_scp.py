from pathlib import Path

import numpy as np
import pytest

from espnet2.fileio.npy_scp import NpyScpReader, NpyScpWriter
from espnet2.fileio.sound_scp import SoundScpReader, SoundScpWriter


def test_NpyScpReader(tmp_path: Path):
    npy_path1 = tmp_path / "a1.npy"
    array1 = np.random.randn(1)
    npy_path2 = tmp_path / "a2.npy"
    array2 = np.random.randn(1, 1, 10)
    np.save(npy_path1, array1)
    np.save(npy_path2, array2)

    p = tmp_path / "dummy.scp"
    with p.open("w") as f:
        f.write(f"abc {npy_path1}\n")
        f.write(f"def {npy_path2}\n")

    desired = {"abc": array1, "def": array2}
    target = NpyScpReader(p)

    for k in desired:
        t = target[k]
        d = desired[k]
        np.testing.assert_array_equal(t, d)

    assert len(target) == len(desired)
    assert "abc" in target
    assert "def" in target
    assert tuple(target.keys()) == tuple(desired)
    assert tuple(target) == tuple(desired)
    assert target.get_path("abc") == str(npy_path1)
    assert target.get_path("def") == str(npy_path2)


def test_NpyScpWriter(tmp_path: Path):
    array1 = np.random.randn(1)
    array2 = np.random.randn(1, 1, 10)
    with NpyScpWriter(tmp_path, tmp_path / "feats.scp") as writer:
        writer["abc"] = array1
        writer["def"] = array2
    target = NpyScpReader(tmp_path / "feats.scp")
    desired = {"abc": array1, "def": array2}

    for k in desired:
        t = target[k]
        d = desired[k]
        np.testing.assert_array_equal(t, d)

    assert writer.get_path("abc") == str(tmp_path / "abc.npy")
    assert writer.get_path("def") == str(tmp_path / "def.npy")


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


def test_SoundScpWriter_normalize(tmp_path: Path):
    audio1 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio2 = np.random.randint(-100, 100, 16, dtype=np.int16)
    audio1 = audio1.astype(np.float64) / (np.iinfo(np.int16).max + 1)
    audio2 = audio2.astype(np.float64) / (np.iinfo(np.int16).max + 1)

    with SoundScpWriter(tmp_path, tmp_path / "wav.scp", dtype=np.int16) as writer:
        writer["abc"] = 16, audio1
        writer["def"] = 16, audio2
        # Unsupported dimension
        with pytest.raises(RuntimeError):
            y = np.random.randint(-100, 100, [16, 1, 1], dtype=np.int16)
            writer["ghi"] = 16, y
    target = SoundScpReader(tmp_path / "wav.scp", normalize=True, dtype=np.float64)
    desired = {"abc": (16, audio1), "def": (16, audio2)}

    for k in desired:
        rate1, t = target[k]
        rate2, d = desired[k]
        assert rate1 == rate2
        np.testing.assert_array_equal(t, d)
