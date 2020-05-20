import h5py
import kaldiio
import numpy as np
from PIL import Image
import pytest
import soundfile

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.train.dataset import ESPnetDataset


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
    dataset = ESPnetDataset(
        path_name_type_list=[(sound_scp, "data1", "sound")], preprocess=preprocess,
    )
    print(dataset)
    print(dataset.names())
    assert len(dataset) == 2
    assert dataset.has_name("data1")

    _, data = dataset["a"]
    assert data["data1"].shape == (160000,)

    _, data = dataset["b"]
    assert data["data1"].shape == (80000,)


@pytest.fixture
def pipe_wav(tmp_path):
    p = tmp_path / "wav.scp"
    soundfile.write(
        tmp_path / "a.wav",
        np.random.randint(-100, 100, (160000,), dtype=np.int16),
        16000,
    )
    soundfile.write(
        tmp_path / "b.wav",
        np.random.randint(-100, 100, (80000,), dtype=np.int16),
        16000,
    )
    with p.open("w") as f:
        f.write(f"a {tmp_path / 'a.wav'}\n")
        f.write(f"b {tmp_path / 'b.wav'}\n")
    return str(p)


def test_ESPnetDataset_pipe_wav(pipe_wav):
    dataset = ESPnetDataset(
        path_name_type_list=[(pipe_wav, "data1", "pipe_wav")], preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert data["data1"].shape == (160000,)

    _, data = dataset["b"]
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
    dataset = ESPnetDataset(
        path_name_type_list=[(feats_scp, "data2", "kaldi_ark")], preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert data["data2"].shape == (100, 80,)

    _, data = dataset["b"]
    assert data["data2"].shape == (150, 80,)


@pytest.fixture
def npy_scp(tmp_path):
    p = tmp_path / "npy.scp"
    w = NpyScpWriter(tmp_path / "data", p)
    w["a"] = np.random.randn(100, 80)
    w["b"] = np.random.randn(150, 80)
    return str(p)


def test_ESPnetDataset_npy_scp(npy_scp):
    dataset = ESPnetDataset(
        path_name_type_list=[(npy_scp, "data3", "npy")], preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert data["data3"].shape == (100, 80,)

    _, data = dataset["b"]
    assert data["data3"].shape == (150, 80,)


@pytest.fixture
def h5file_1(tmp_path):
    p = tmp_path / "file.h5"
    with h5py.File(p, "w") as w:
        w["a"] = np.random.randn(100, 80)
        w["b"] = np.random.randn(150, 80)
    return str(p)


@pytest.fixture
def h5file_2(tmp_path):
    p = tmp_path / "file.h5"
    with h5py.File(p, "w") as w:
        w["a/input"] = np.random.randn(100, 80)
        w["a/target"] = np.random.randint(0, 10, (10,))
        w["b/input"] = np.random.randn(150, 80)
        w["b/target"] = np.random.randint(0, 10, (13,))
    return str(p)


def test_ESPnetDataset_h5file_1(h5file_1):
    dataset = ESPnetDataset(
        path_name_type_list=[(h5file_1, "data4", "hdf5")], preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert data["data4"].shape == (100, 80,)

    _, data = dataset["b"]
    assert data["data4"].shape == (150, 80,)


def test_ESPnetDataset_h5file_2(h5file_2):
    dataset = ESPnetDataset(
        path_name_type_list=[(h5file_2, "data1", "hdf5")], preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert data["data1_input"].shape == (100, 80)
    assert data["data1_target"].shape == (10,)

    _, data = dataset["b"]
    assert data["data1_input"].shape == (150, 80)
    assert data["data1_target"].shape == (13,)


@pytest.fixture
def shape_file(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 100,80\n")
        f.write("b 150,80\n")
    return str(p)


def test_ESPnetDataset_rand_float(shape_file):
    dataset = ESPnetDataset(
        path_name_type_list=[(shape_file, "data5", "rand_float")],
        preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert data["data5"].shape == (100, 80,)

    _, data = dataset["b"]
    assert data["data5"].shape == (150, 80,)


def test_ESPnetDataset_rand_int(shape_file):
    dataset = ESPnetDataset(
        path_name_type_list=[(shape_file, "data6", "rand_int_0_10")],
        preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert data["data6"].shape == (100, 80,)

    _, data = dataset["b"]
    assert data["data6"].shape == (150, 80,)


@pytest.fixture
def text(tmp_path):
    p = tmp_path / "text"
    with p.open("w") as f:
        f.write("a hello world\n")
        f.write("b foo bar\n")
    return str(p)


def test_ESPnetDataset_text(text):
    dataset = ESPnetDataset(
        path_name_type_list=[(text, "data7", "text")], preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert tuple(data["data7"]) == (0,)

    _, data = dataset["b"]
    assert tuple(data["data7"]) == (1,)


@pytest.fixture
def text_float(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 1.4 3.4\n")
        f.write("b 0.9 9.3\n")
    return str(p)


def test_ESPnetDataset_text_float(text_float):
    dataset = ESPnetDataset(
        path_name_type_list=[(text_float, "data8", "text_float")],
        preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert all((data["data8"]) == np.array([1.4, 3.4], dtype=np.float32))

    _, data = dataset["b"]
    assert all((data["data8"]) == np.array([0.9, 9.3], dtype=np.float32))


@pytest.fixture
def text_int(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 0 1 2\n")
        f.write("b 2 3 4\n")
    return str(p)


def test_ESPnetDataset_text_int(text_int):
    dataset = ESPnetDataset(
        path_name_type_list=[(text_int, "data8", "text_int")], preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert tuple(data["data8"]) == (0, 1, 2)

    _, data = dataset["b"]
    assert tuple(data["data8"]) == (2, 3, 4)


@pytest.fixture
def csv_float(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 1.4,3.4\n")
        f.write("b 0.9,9.3\n")
    return str(p)


def test_ESPnetDataset_csv_float(csv_float):
    dataset = ESPnetDataset(
        path_name_type_list=[(csv_float, "data8", "csv_float")], preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert all((data["data8"]) == np.array([1.4, 3.4], dtype=np.float32))

    _, data = dataset["b"]
    assert all((data["data8"]) == np.array([0.9, 9.3], dtype=np.float32))


@pytest.fixture
def csv_int(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        f.write("a 0,1,2\n")
        f.write("b 2,3,4\n")
    return str(p)


def test_ESPnetDataset_csv_int(csv_int):
    dataset = ESPnetDataset(
        path_name_type_list=[(csv_int, "data8", "csv_int")], preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert tuple(data["data8"]) == (0, 1, 2)

    _, data = dataset["b"]
    assert tuple(data["data8"]) == (2, 3, 4)


@pytest.fixture
def imagefolder(tmp_path):
    p = tmp_path / "img"
    (p / "a").mkdir(parents=True)
    (p / "b").mkdir(parents=True)
    a = np.random.rand(30, 30, 3) * 255
    im_out = Image.fromarray(a.astype("uint8")).convert("RGBA")
    im_out.save(p / "a" / "foo.png")
    a = np.random.rand(30, 30, 3) * 255
    im_out = Image.fromarray(a.astype("uint8")).convert("RGBA")
    im_out.save(p / "b" / "foo.png")
    return str(p)


def test_ESPnetDataset_imagefolder(imagefolder):
    pytest.importorskip("torchvision")

    dataset = ESPnetDataset(
        path_name_type_list=[(imagefolder, "data1", "imagefolder_32x32")],
        preprocess=preprocess,
    )

    _, data = dataset[0]
    assert data["data1_0"].shape == (3, 32, 32)
    assert data["data1_1"] == (0,)
    _, data = dataset[1]
    assert data["data1_0"].shape == (3, 32, 32)
    assert data["data1_1"] == (1,)
