import h5py
import pytest

from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler


@pytest.fixture()
def shape_files(tmp_path):
    p1 = tmp_path / "shape1.txt"
    with p1.open("w") as f:
        f.write("a 1000,80\n")
        f.write("b 400,80\n")
        f.write("c 800,80\n")
        f.write("d 789,80\n")
        f.write("e 1023,80\n")
        f.write("f 999,80\n")

    p2 = tmp_path / "shape2.txt"
    with p2.open("w") as f:
        f.write("a 30,30\n")
        f.write("b 50,30\n")
        f.write("c 39,30\n")
        f.write("d 49,30\n")
        f.write("e 44,30\n")
        f.write("f 99,30\n")

    return str(p1), str(p2)


@pytest.fixture()
def shape_files_hdf5(tmp_path):
    p = tmp_path / "corpus.h5"
    f = h5py.File(p, "a")
    f["0/a"] = (1000, 80)
    f["0/b"] = (400, 80)
    f["0/c"] = (800, 80)
    f["0/d"] = (789, 80)
    f["0/e"] = (1023, 80)
    f["0/f"] = (999, 80)

    f["1/a"] = (30, 30)
    f["1/b"] = (50, 30)
    f["1/c"] = (39, 30)
    f["1/d"] = (49, 30)
    f["1/e"] = (44, 30)
    f["1/f"] = (99, 30)

    return f["0"], f["1"]


@pytest.mark.parametrize("drop_last", [True, False])
def test_UnsortedBatchSampler(shape_files, drop_last):
    sampler = UnsortedBatchSampler(2, key_file=shape_files[0], drop_last=drop_last)
    list(sampler)


@pytest.mark.parametrize("drop_last", [True, False])
def test_UnsortedBatchSampler_hdf5(shape_files_hdf5, drop_last):
    sampler = UnsortedBatchSampler(2, key_file=shape_files_hdf5[0], drop_last=drop_last)
    list(sampler)


@pytest.mark.parametrize("drop_last", [True, False])
def test_UnsortedBatchSampler_repr(shape_files, drop_last):
    sampler = UnsortedBatchSampler(2, key_file=shape_files[0], drop_last=drop_last)
    print(sampler)


@pytest.mark.parametrize("drop_last", [True, False])
def test_UnsortedBatchSampler_len(shape_files, drop_last):
    sampler = UnsortedBatchSampler(2, key_file=shape_files[0], drop_last=drop_last)
    len(sampler)
