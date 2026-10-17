import pytest

from espnet2.samplers.category_balanced_sampler import CategoryBalancedSampler


@pytest.fixture()
def category_file(tmp_path):
    p = tmp_path / "category2utt"
    with p.open("w") as f:
        f.write("c1 a b c d\n")
        f.write("c2 e f g\n")
        f.write("c3 m n o p q")

    return str(p)


@pytest.mark.parametrize("drop_last", [True, False])
def test_CategoryBalancedSampler(category_file, drop_last):
    sampler = CategoryBalancedSampler(
        2,
        category2utt_file=category_file,
        drop_last=drop_last,
    )
    list(sampler)


@pytest.mark.parametrize("drop_last", [True, False])
def test_CategoryBalancedSampler_repr(category_file, drop_last):
    sampler = CategoryBalancedSampler(
        2,
        category2utt_file=category_file,
        drop_last=drop_last,
    )
    print(sampler)


@pytest.mark.parametrize("drop_last", [True, False])
def test_CategoryBalancedSampler_len(category_file, drop_last):
    sampler = CategoryBalancedSampler(
        2,
        category2utt_file=category_file,
        drop_last=drop_last,
    )
    len(sampler)
