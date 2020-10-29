import pytest

from espnet2.samplers.build_batch_sampler import build_batch_sampler


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


@pytest.mark.parametrize(
    "type", ["unsorted", "sorted", "folded", "length", "numel", "foo"]
)
def test_build_batch_sampler(shape_files, type):
    if type == "foo":
        with pytest.raises(ValueError):
            build_batch_sampler(
                batch_bins=60000,
                batch_size=2,
                shape_files=shape_files,
                fold_lengths=[800, 40],
                type=type,
            )
    else:
        sampler = build_batch_sampler(
            batch_bins=60000,
            batch_size=2,
            shape_files=shape_files,
            fold_lengths=[800, 40],
            type=type,
        )
        list(sampler)


def test_build_batch_sampler_invalid_fold_lengths(shape_files):
    with pytest.raises(ValueError):
        build_batch_sampler(
            batch_bins=60000,
            batch_size=2,
            shape_files=shape_files,
            fold_lengths=[800, 40, 100],
            type="seq",
        )
