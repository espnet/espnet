import h5py
import kaldiio
import numpy as np
import pytest

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.train.iterable_dataset import IterableESPnetDataset


def preprocess(id: str, data):
    new_data = {}
    for k, v in data.items():
        if isinstance(v, str):
            if v == "hello world":
                new_data[k] = np.array([0])
            elif v == "foo bar":
                new_data[k] = np.array([1])
            else:
                new_data[k] = np.array([2])
        else:
            new_data[k] = v
    return new_data


@pytest.fixture
def sound_scp(tmp_path):
    p = tmp_path / "wav.scp"
    w = SoundScpWriter(tmp_path / "data", p)
    w["a"] = 16000, np.random.randint(-100, 100, (160000,), dtype=np.int16)
    w["b"] = 16000, np.random.randint(-100, 100, (80000,), dtype=np.int16)
    return str(p)


def test_ESPnetDataset_sound_scp(sound_scp):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(sound_scp, "data1", "sound")], preprocess=preprocess,
    )
    print(dataset)
    print(dataset.names())
    assert dataset.has_name("data1")

    for key, data in dataset:
        if key == "a":
            assert data["data1"].shape == (160000,)
        if key == "b":
            assert data["data1"].shape == (80000,)


@pytest.fixture
def feats_scp(tmp_path):
    p = tmp_path / "feats.scp"
    p2 = tmp_path / "feats.ark"
    with kaldiio.WriteHelper(f"ark,scp:{p2},{p}") as w:
        w["a"] = np.random.randn(100, 80)
        w["b"] = np.random.randn(150, 80)
    return str(p)


def test_ESPnetDataset_feats_scp(feats_scp,):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(feats_scp, "data2", "kaldi_ark")], preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert data["data2"].shape == (100, 80,)
        if key == "b":
            assert data["data2"].shape == (150, 80,)


@pytest.fixture
def npy_scp(tmp_path):
    p = tmp_path / "npy.scp"
    w = NpyScpWriter(tmp_path / "data", p)
    w["a"] = np.random.randn(100, 80)
    w["b"] = np.random.randn(150, 80)
    return str(p)


def test_ESPnetDataset_npy_scp(npy_scp):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(npy_scp, "data3", "npy")], preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert data["data3"].shape == (100, 80,)
        if key == "b":
            assert data["data3"].shape == (150, 80,)


@pytest.fixture
def h5file_1(tmp_path):
    p = tmp_path / "file.h5"
    with h5py.File(p, "w") as w:
        w["a"] = np.random.randn(100, 80)
        w["b"] = np.random.randn(150, 80)
    return str(p)


def test_ESPnetDataset_h5file_1(h5file_1):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(h5file_1, "data4", "hdf5")], preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert data["data4"].shape == (100, 80,)
        if key == "b":
            assert data["data4"].shape == (150, 80,)


@pytest.fixture
def shape_file(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 100,80\n")
        f.write("b 150,80\n")
    return str(p)


def test_ESPnetDataset_rand_float(shape_file):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(shape_file, "data5", "rand_float")],
        preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert data["data5"].shape == (100, 80,)
        if key == "b":
            assert data["data5"].shape == (150, 80,)


def test_ESPnetDataset_rand_int(shape_file):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(shape_file, "data6", "rand_int_0_10")],
        preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert data["data6"].shape == (100, 80,)
        if key == "b":
            assert data["data6"].shape == (150, 80,)


@pytest.fixture
def text(tmp_path):
    p = tmp_path / "text"
    with p.open("w") as f:
        f.write("a hello world\n")
        f.write("b foo bar\n")
    return str(p)


def test_ESPnetDataset_text(text):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(text, "data7", "text")], preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert tuple(data["data7"]) == (0,)
        if key == "b":
            assert tuple(data["data7"]) == (1,)


@pytest.fixture
def text_float(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 1.4 3.4\n")
        f.write("b 0.9 9.3\n")
    return str(p)


def test_ESPnetDataset_text_float(text_float):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(text_float, "data8", "text_float")],
        preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert all((data["data8"]) == np.array([1.4, 3.4], dtype=np.float32))
        if key == "b":
            assert all((data["data8"]) == np.array([0.9, 9.3], dtype=np.float32))


@pytest.fixture
def text_int(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 0 1 2\n")
        f.write("b 2 3 4\n")
    return str(p)


def test_ESPnetDataset_text_int(text_int):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(text_int, "data8", "text_int")], preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert tuple(data["data8"]) == (0, 1, 2)
        if key == "b":
            assert tuple(data["data8"]) == (2, 3, 4)


@pytest.fixture
def csv_float(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 1.4,3.4\n")
        f.write("b 0.9,9.3\n")
    return str(p)


def test_ESPnetDataset_csv_float(csv_float):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(csv_float, "data8", "csv_float")], preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert all((data["data8"]) == np.array([1.4, 3.4], dtype=np.float32))
        if key == "b":
            assert all((data["data8"]) == np.array([0.9, 9.3], dtype=np.float32))


@pytest.fixture
def csv_int(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 0,1,2\n")
        f.write("b 2,3,4\n")
    return str(p)


def test_ESPnetDataset_csv_int(csv_int):
    dataset = IterableESPnetDataset(
        path_name_type_list=[(csv_int, "data8", "csv_int")], preprocess=preprocess,
    )

    for key, data in dataset:
        if key == "a":
            assert tuple(data["data8"]) == (0, 1, 2)
        if key == "b":
            assert tuple(data["data8"]) == (2, 3, 4)
