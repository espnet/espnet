import pytest

from espnet2.samplers.weighted_batch_sampler import WeightedBatchSampler


@pytest.fixture()
def utt2weight_file(tmp_path):
    p = tmp_path / "utt2weight.txt"
    with p.open("w") as f:
        f.write("a 0.1\n")
        f.write("b 0.2\n")
        f.write("c 0.3\n")
        f.write("d 0.4\n")
        f.write("e 0.5\n")
        f.write("f 0.6\n")
    return str(p)


def test_WeightedBatchSampler(utt2weight_file):
    sampler = WeightedBatchSampler(
        2,
        utt2weight_file=utt2weight_file,
    )
    list(sampler)


def test_WeightedBatchSampler_repr(utt2weight_file):
    sampler = WeightedBatchSampler(
        2,
        utt2weight_file=utt2weight_file,
    )
    print(sampler)


def test_WeightedBatchSampler_len(utt2weight_file):
    sampler = WeightedBatchSampler(
        2,
        utt2weight_file=utt2weight_file,
    )
    len(sampler)


def test_WeightedBatchSampler_generate(utt2weight_file):
    sampler = WeightedBatchSampler(
        2,
        utt2weight_file=utt2weight_file,
    )
    samples = sampler.generate(1)
    print(samples)
