import pytest

from espnet2.samplers.category_power_sampler import (
    CategoryDatasetPowerSampler,
    CategoryPowerSampler,
)


@pytest.fixture()
def category2utt_file(tmp_path):
    p = tmp_path / "category2utt"
    with p.open("w") as f:
        f.write("c1 utt1 utt2 utt3\n")
        f.write("c2 utt4 utt5 utt6\n")
        f.write("c3 utt7 utt8 utt9")

    return str(p)


@pytest.fixture()
def dataset2utt_file(tmp_path):
    p = tmp_path / "dataset2utt"
    with p.open("w") as f:
        f.write("d1 utt1 utt2 utt3\n")
        f.write("d2 utt4 utt5 utt6 utt7 utt8\n")
        f.write("d3 utt9")

    return str(p)


@pytest.fixture()
def utt2dataset_file(tmp_path):
    p = tmp_path / "utt2dataset"
    with p.open("w") as f:
        f.write("utt1 d1\n")
        f.write("utt2 d1\n")
        f.write("utt3 d1\n")
        f.write("utt4 d2\n")
        f.write("utt5 d2\n")
        f.write("utt6 d2\n")
        f.write("utt7 d2\n")
        f.write("utt8 d2\n")
        f.write("utt9 d3")

    return str(p)


@pytest.fixture()
def shape_file(tmp_path):
    p = tmp_path / "speech_shape"
    with p.open("w") as f:
        f.write("utt1 500\n")
        f.write("utt2 600\n")
        f.write("utt3 400\n")
        f.write("utt4 500\n")
        f.write("utt5 500\n")
        f.write("utt6 550\n")
        f.write("utt7 700\n")
        f.write("utt8 800\n")
        f.write("utt9 900")

    return str(p)


@pytest.mark.parametrize("drop_last", [True, False])
def test_CategoryPowerSampler(category2utt_file, shape_file, drop_last):
    sampler = CategoryPowerSampler(
        batch_bins=1500,
        shape_files=[shape_file],
        min_batch_size=1,
        max_batch_size=None,
        upsampling_factor=0.5,
        dataset_scaling_factor=1.2,
        drop_last=drop_last,
        category2utt_file=category2utt_file,
        epoch=1,
    )

    assert hasattr(sampler, "__iter__")

    batches = list(sampler)
    assert len(batches) > 0


@pytest.mark.parametrize("drop_last", [True, False])
def test_CategoryDatasetPowerSampler(
    category2utt_file, dataset2utt_file, utt2dataset_file, shape_file, drop_last
):
    sampler = CategoryDatasetPowerSampler(
        batch_bins=1500,
        shape_files=[shape_file],
        min_batch_size=1,
        max_batch_size=None,
        category_upsampling_factor=0.5,
        dataset_upsampling_factor=0.5,
        dataset_scaling_factor=1.2,
        drop_last=drop_last,
        category2utt_file=category2utt_file,
        dataset2utt_file=dataset2utt_file,
        utt2dataset_file=utt2dataset_file,
        epoch=1,
    )

    assert hasattr(sampler, "__iter__")

    batches = list(sampler)
    assert len(batches) > 0
