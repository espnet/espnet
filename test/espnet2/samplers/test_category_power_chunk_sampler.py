from collections import Counter

import pytest

from espnet2.samplers.category_power_chunk_sampler import CategoryPowerChunkSampler


def _chunks():
    # category a: 90 chunks over two sequences, b: 9, c: 1
    chunks = [("a1", i, i + 10) for i in range(0, 600, 10)]
    chunks += [("a2", i, i + 10) for i in range(0, 300, 10)]
    chunks += [("b1", i, i + 10) for i in range(0, 90, 10)]
    chunks += [("c1", 0, 10)]
    return chunks


UTT2CAT = {"a1": "a", "a2": "a", "b1": "b", "c1": "c"}


def _per_category(batches):
    return Counter(UTT2CAT[uid] for b in batches for uid, _, _ in b)


def test_factor_one_keeps_every_chunk_once():
    s = CategoryPowerChunkSampler(_chunks(), UTT2CAT, 8, 1.0)
    drawn = [c for b in s.generate(3) for c in b]
    assert sorted(drawn) == sorted(_chunks())


def test_factor_zero_balances_the_categories():
    s = CategoryPowerChunkSampler(_chunks(), UTT2CAT, 8, 0.0)
    counts = _per_category(s.generate(1))
    assert sum(counts.values()) == 100
    assert sorted(counts.values()) == [33, 33, 34]
    # the single chunk of c is repeated, a is a subset without repeats
    a = [c for b in s.generate(1) for c in b if UTT2CAT[c[0]] == "a"]
    assert len(set(a)) == len(a)


def test_power_law_shares():
    s = CategoryPowerChunkSampler(_chunks(), UTT2CAT, 8, 0.5)
    w = {"a": 90**0.5, "b": 9**0.5, "c": 1.0}
    total = sum(w.values())
    counts = s.category_counts()
    for c, (n, m) in counts.items():
        assert abs(m - 100 * w[c] / total) < 1
    assert sum(m for _, m in counts.values()) == 100


def test_epochs_differ_and_repeat():
    s = CategoryPowerChunkSampler(_chunks(), UTT2CAT, 8, 0.3)
    assert s.generate(1) == s.generate(1)
    assert s.generate(1) != s.generate(2)
    # downsampled category: other chunks in another epoch
    a = lambda e: {c for b in s.generate(e) for c in b if UTT2CAT[c[0]] == "a"}  # noqa
    assert a(1) != a(2)


def test_batches_split_like_unsorted():
    s = CategoryPowerChunkSampler(_chunks(), UTT2CAT, 8, 0.0)
    batches = s.generate(0)
    assert len(batches) == len(s) == 100 // 8
    assert all(8 <= len(b) <= 9 for b in batches)
    assert (
        len(CategoryPowerChunkSampler(_chunks(), UTT2CAT, 8, 0.0, num_batches=3)) == 3
    )


def test_ranks_share_the_draw():
    full = CategoryPowerChunkSampler(_chunks(), UTT2CAT, 8, 0.0).generate(5)
    parts = [
        CategoryPowerChunkSampler(
            _chunks(), UTT2CAT, 8, 0.0, rank=r, world_size=2
        ).generate(5)
        for r in range(2)
    ]
    for k, b in enumerate(full):
        assert sorted(parts[0][k] + parts[1][k]) == sorted(b)


@pytest.mark.parametrize("beta", [-0.1, 1.5])
def test_bad_factor(beta):
    with pytest.raises(ValueError):
        CategoryPowerChunkSampler(_chunks(), UTT2CAT, 8, beta)


def test_sequence_without_category():
    with pytest.raises(KeyError):
        CategoryPowerChunkSampler(_chunks() + [("z", 0, 10)], UTT2CAT, 8, 0.5)
