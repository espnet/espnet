import pytest

from espnet2.samplers.length_weighted_batch_sampler import WeightedLengthBatchSampler


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


def test_WeightedLengthBatchSampler(utt2weight_file):
    sampler = WeightedLengthBatchSampler(
        2,
        utt2weight_file=utt2weight_file,
    )
    list(sampler)


def test_WeightedLengthBatchSampler_repr(utt2weight_file):
    sampler = WeightedLengthBatchSampler(
        2,
        utt2weight_file=utt2weight_file,
    )
    print(sampler)


def test_WeightedLengthBatchSampler_len(utt2weight_file):
    sampler = WeightedLengthBatchSampler(
        2,
        utt2weight_file=utt2weight_file,
    )
    len(sampler)


def test_WeightedLengthBatchSampler_generate(utt2weight_file):
    sampler = WeightedLengthBatchSampler(
        2,
        utt2weight_file=utt2weight_file,
    )
    samples = sampler.generate(1)
    print(samples)
