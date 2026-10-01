"""Tests for the edit-distance alignment."""

import pytest

from splet.alignment import levenshtein_alignment


def test_identical_sequences_are_all_hits():
    alignment = levenshtein_alignment(["a", "b", "c"], ["a", "b", "c"])
    assert (alignment.hits, alignment.errors) == (3, 0)
    assert alignment.error_rate == 0.0


@pytest.mark.parametrize(
    "ref, hyp, substitutions, deletions, insertions",
    [
        (["a", "b", "c"], ["a", "x", "c"], 1, 0, 0),
        (["a", "b", "c"], ["a", "c"], 0, 1, 0),
        (["a", "c"], ["a", "b", "c"], 0, 0, 1),
        (["a", "b", "c"], ["x", "y", "z"], 3, 0, 0),
    ],
)
def test_counts(ref, hyp, substitutions, deletions, insertions):
    alignment = levenshtein_alignment(ref, hyp)
    assert alignment.substitutions == substitutions
    assert alignment.deletions == deletions
    assert alignment.insertions == insertions


@pytest.mark.parametrize(
    "ref, hyp, rate",
    [
        ([], [], 0.0),
        # An empty reference cannot produce a meaningful denominator. Both
        # branches are pinned here so that a future change to the convention
        # is a test failure rather than a silent shift in every corpus that
        # contains an empty reference.
        ([], ["a"], 1.0),
        (["a"], [], 1.0),
    ],
)
def test_empty_sequences(ref, hyp, rate):
    assert levenshtein_alignment(ref, hyp).error_rate == rate


def test_operations_reconstruct_both_sides():
    ref = "the quick brown fox".split()
    hyp = "the quik brown fox now".split()
    alignment = levenshtein_alignment(ref, hyp)
    assert [r for _, r, _ in alignment.operations if r is not None] == ref
    assert [h for _, _, h in alignment.operations if h is not None] == hyp


def test_counts_are_consistent_with_lengths():
    ref = "a b c d e".split()
    hyp = "a x c e f".split()
    alignment = levenshtein_alignment(ref, hyp)
    assert alignment.hits + alignment.substitutions + alignment.deletions == len(ref)
    assert alignment.hits + alignment.substitutions + alignment.insertions == len(hyp)


def test_python_backend_refuses_an_oversized_table():
    with pytest.raises(MemoryError, match="rapidfuzz"):
        levenshtein_alignment(["w"] * 3000, ["w"] * 3000, backend="python")


def test_rapidfuzz_refuses_a_weighted_cost_model():
    """rapidfuzz computes unit-cost edits, so it cannot answer for sclite's.

    Returning its unit-cost split under ``costs="sclite"`` would be a wrong
    answer rather than a slower one, so the combination is refused.
    """
    pytest.importorskip("rapidfuzz")
    with pytest.raises(ValueError, match="unit-cost"):
        levenshtein_alignment(["a"], ["b"], backend="rapidfuzz", costs="sclite")


def test_backends_agree_on_totals():
    """The two backends must agree on the error count and the rate.

    Under unit costs, which is the only model rapidfuzz can express. They are
    not required to agree on the S/D/I split, and they do not: when several
    alignments tie at the minimum cost, each breaks the tie its own way.
    """
    pytest.importorskip("rapidfuzz")
    import random

    random.seed(0)
    vocabulary = "a b c d e".split()
    for _ in range(50):
        ref = [random.choice(vocabulary) for _ in range(random.randint(0, 12))]
        hyp = [random.choice(vocabulary) for _ in range(random.randint(0, 12))]
        python = levenshtein_alignment(ref, hyp, backend="python", costs="unit")
        rapid = levenshtein_alignment(ref, hyp, backend="rapidfuzz", costs="unit")
        assert python.errors == rapid.errors, f"ref={ref} hyp={hyp}"
        assert python.error_rate == rapid.error_rate, f"ref={ref} hyp={hyp}"


def test_the_split_is_not_backend_independent():
    """Pin the disagreement, so that it is a documented fact and not a
    surprise found later in a results table.

    This is a property of the unit-cost model only. The default model is
    sclite's, where the alternatives rarely tie and the python backend is the
    only one that implements it, so the default path does not depend on which
    packages happen to be installed.
    """
    pytest.importorskip("rapidfuzz")
    ref = "d a c e d d".split()
    hyp = "c d c e b e b c b a e c".split()
    python = levenshtein_alignment(ref, hyp, backend="python", costs="unit")
    rapid = levenshtein_alignment(ref, hyp, backend="rapidfuzz", costs="unit")
    assert python.errors == rapid.errors
    assert (python.substitutions, python.deletions, python.insertions) != (
        rapid.substitutions,
        rapid.deletions,
        rapid.insertions,
    )
