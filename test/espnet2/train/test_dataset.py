from pathlib import Path

from espnet2.train.dataset import ESPnetDataset


def preprocess(id: str, data):
    return data


def test_dataset_with_rand(tmp_path: Path):
    with (tmp_path / "float_shape.txt").open("w") as f:
        f.write("a 30,80\n")
        f.write("b 40,80\n")
    with (tmp_path / "int_shape.txt").open("w") as f:
        f.write("a 10\n")
        f.write("b 20\n")

    dataset = ESPnetDataset(
        path_name_type_list=[
            (str(tmp_path / "float_shape.txt"), "data1", "rand_float"),
            (str(tmp_path / "int_shape.txt"), "data2", "rand_int_0_10"),
        ],
        preprocess=preprocess,
    )

    _, data = dataset["a"]
    assert data["data1"].shape == (30, 80)
    assert data["data2"].shape == (10,)

    _, data = dataset["b"]
    assert data["data1"].shape == (40, 80)
    assert data["data2"].shape == (20,)
