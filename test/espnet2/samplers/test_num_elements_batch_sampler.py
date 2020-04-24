import h5py
import pytest

from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler


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


@pytest.mark.parametrize("sort_in_batch", ["descending", "ascending"])
@pytest.mark.parametrize("sort_batch", ["descending", "ascending"])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("padding", [True, False])
def test_NumElementsBatchSampler(
    shape_files, sort_in_batch, sort_batch, drop_last, padding,
):
    sampler = NumElementsBatchSampler(
        60000,
        shape_files=shape_files,
        sort_in_batch=sort_in_batch,
        sort_batch=sort_batch,
        drop_last=drop_last,
        padding=padding,
    )
    list(sampler)


@pytest.mark.parametrize("sort_in_batch", ["descending", "ascending"])
@pytest.mark.parametrize("sort_batch", ["descending", "ascending"])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("padding", [True, False])
def test_NumElementsBatchSampler_hdf5(
    shape_files_hdf5, sort_in_batch, sort_batch, drop_last, padding,
):
    sampler = NumElementsBatchSampler(
        60000,
        shape_files=shape_files_hdf5,
        sort_in_batch=sort_in_batch,
        sort_batch=sort_batch,
        drop_last=drop_last,
        padding=padding,
    )
    list(sampler)


@pytest.mark.parametrize("sort_in_batch", ["descending", "ascending"])
@pytest.mark.parametrize("sort_batch", ["descending", "ascending"])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("padding", [True, False])
def test_NumElementsBatchSampler_repr(
    shape_files, sort_in_batch, sort_batch, drop_last, padding
):
    sampler = NumElementsBatchSampler(
        60000,
        shape_files=shape_files,
        sort_in_batch=sort_in_batch,
        sort_batch=sort_batch,
        drop_last=drop_last,
        padding=padding,
    )
    print(sampler)


@pytest.mark.parametrize("sort_in_batch", ["descending", "ascending"])
@pytest.mark.parametrize("sort_batch", ["descending", "ascending"])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("padding", [True, False])
def test_NumElementsBatchSampler_len(
    shape_files, sort_in_batch, sort_batch, drop_last, padding
):
    sampler = NumElementsBatchSampler(
        60000,
        shape_files=shape_files,
        sort_in_batch=sort_in_batch,
        sort_batch=sort_batch,
        drop_last=drop_last,
        padding=padding,
    )
    len(sampler)
