import pytest

from espnet2.samplers.folded_batch_sampler import FoldedBatchSampler


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


@pytest.mark.parametrize("sort_in_batch", ["descending", "ascending"])
@pytest.mark.parametrize("sort_batch", ["descending", "ascending"])
@pytest.mark.parametrize("drop_last", [True, False])
def test_FoldedBatchSampler(shape_files, sort_in_batch, sort_batch, drop_last):
    sampler = FoldedBatchSampler(
        2,
        shape_files=shape_files,
        fold_lengths=[500, 80],
        sort_in_batch=sort_in_batch,
        sort_batch=sort_batch,
        drop_last=drop_last,
    )
    list(sampler)


@pytest.mark.parametrize("sort_in_batch", ["descending", "ascending"])
@pytest.mark.parametrize("sort_batch", ["descending", "ascending"])
@pytest.mark.parametrize("drop_last", [True, False])
def test_FoldedBatchSampler_repr(shape_files, sort_in_batch, sort_batch, drop_last):
    sampler = FoldedBatchSampler(
        2,
        shape_files=shape_files,
        fold_lengths=[500, 80],
        sort_in_batch=sort_in_batch,
        sort_batch=sort_batch,
        drop_last=drop_last,
    )
    print(sampler)


@pytest.mark.parametrize("sort_in_batch", ["descending", "ascending"])
@pytest.mark.parametrize("sort_batch", ["descending", "ascending"])
@pytest.mark.parametrize("drop_last", [True, False])
def test_FoldedBatchSampler_len(shape_files, sort_in_batch, sort_batch, drop_last):
    sampler = FoldedBatchSampler(
        2,
        shape_files=shape_files,
        fold_lengths=[500, 80],
        sort_in_batch=sort_in_batch,
        sort_batch=sort_batch,
        drop_last=drop_last,
    )
    len(sampler)


@pytest.fixture()
def five_shape_file(tmp_path):
    p = tmp_path / "shape.txt"
    with p.open("w") as f:
        for i, k in enumerate("abcde", 1):
            f.write(f"{k} {i * 10},80\n")
    return str(p)


@pytest.mark.parametrize("drop_last", [True, False])
def test_FoldedBatchSampler_drop_last(five_shape_file, drop_last):
    sampler = FoldedBatchSampler(
        2,
        shape_files=[five_shape_file],
        fold_lengths=[1000],
        sort_in_batch="ascending",
        drop_last=drop_last,
    )
    expected = [("a", "b"), ("c", "d")]
    if not drop_last:
        expected.append(("e",))
    assert list(sampler) == expected


def test_FoldedBatchSampler_drop_last_keeps_small_category(five_shape_file, tmp_path):
    utt2category = tmp_path / "utt2category"
    with utt2category.open("w") as f:
        f.write("a x\nb x\nc x\nd x\ne y\n")
    sampler = FoldedBatchSampler(
        2,
        shape_files=[five_shape_file],
        fold_lengths=[1000],
        sort_in_batch="ascending",
        drop_last=True,
        utt2category_file=str(utt2category),
    )
    assert list(sampler) == [("a", "b"), ("c", "d"), ("e",)]


@pytest.mark.parametrize(
    "batch_size, min_batch_size, expected",
    [
        (4, 4, [("a", "b", "c", "d", "e")]),
        (2, 3, [("a", "b", "c", "d", "e")]),
    ],
)
def test_FoldedBatchSampler_min_batch_size(
    five_shape_file, batch_size, min_batch_size, expected
):
    sampler = FoldedBatchSampler(
        batch_size,
        shape_files=[five_shape_file],
        fold_lengths=[1000],
        min_batch_size=min_batch_size,
        sort_in_batch="ascending",
    )
    assert list(sampler) == expected
