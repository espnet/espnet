from pathlib import Path

import numpy as np
import pytest
import soundfile

from espnet2.utils.fileio import DatadirWriter
from espnet2.utils.fileio import load_num_sequence_text
from espnet2.utils.fileio import NpyScpReader
from espnet2.utils.fileio import NpyScpWriter
from espnet2.utils.fileio import read_2column_text
from espnet2.utils.fileio import SoundScpReader
from espnet2.utils.fileio import SoundScpWriter


def test_read_2column_text(tmp_path: Path):
    p = tmp_path / "dummy.scp"
    with p.open("w") as f:
        f.write("abc /some/path/a.wav\n")
        f.write("def /some/path/b.wav\n")
    d = read_2column_text(p)
    assert d == {"abc": "/some/path/a.wav", "def": "/some/path/b.wav"}


@pytest.mark.parametrize(
    "loader_type", ["text_int", "text_float", "csv_int", "csv_float", "dummy"]
)
def test_load_num_sequence_text(loader_type: str, tmp_path: Path):
    p = tmp_path / "dummy.txt"
    if "csv" in loader_type:
        delimiter = ","
    else:
        delimiter = " "

    with p.open("w") as f:
        f.write("abc " + delimiter.join(["0", "1", "2"]) + "\n")
        f.write("def " + delimiter.join(["3", "4", "5"]) + "\n")
    desired = {"abc": np.array([0, 1, 2]), "def": np.array([3, 4, 5])}
    if loader_type == "dummy":
        with pytest.raises(ValueError):
            load_num_sequence_text(p, loader_type=loader_type)
        return
    else:
        target = load_num_sequence_text(p, loader_type=loader_type)
    for k in desired:
        np.testing.assert_array_equal(target[k], desired[k])


def test_load_num_sequence_text_invalid(tmp_path: Path):
    p = tmp_path / "dummy.txt"
    with p.open("w") as f:
        f.write("abc 12.3.3.,4.44\n")
    with pytest.raises(ValueError):
        load_num_sequence_text(p)

    with p.open("w") as f:
        f.write("abc\n")
    with pytest.raises(RuntimeError):
        load_num_sequence_text(p)

    with p.open("w") as f:
        f.write("abc 1 2\n")
        f.write("abc 2 4\n")
    with pytest.raises(RuntimeError):
        load_num_sequence_text(p)


def test_DatadirWriter(tmp_path: Path):
    writer = DatadirWriter(tmp_path)
    # enter(), __exit__(), close()
    with writer as f:
        # __getitem__()
        sub = f["aa"]
        # __setitem__()
        sub["bb"] = "aa"

        with pytest.raises(TypeError):
            sub["bb"] = 1
        with pytest.raises(RuntimeError):
            # Already has children
            f["aa"] = "dd"
        with pytest.raises(RuntimeError):
            # Is a text
            sub["cc"]

        # Create a directory, but set mismatched ids
        f["aa2"]["ccccc"] = "aaa"
        # Duplicated warning
        f["aa2"]["ccccc"] = "def"


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
