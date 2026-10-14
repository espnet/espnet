"""Tests for the edit-distance alignment."""

import pytest

from splet.alignment import COST_CHOICES, CostModel, levenshtein_alignment


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


def test_backends_agree_on_totals():
    """The two backends must agree on the error count and the rate.

    Under unit costs, which is all rapidfuzz can compute. They are not
    required to agree on the S/D/I split, and they do not: when several
    alignments tie at the minimum cost, each breaks the tie its own way.
    That is why the default backend is the reference implementation
    rather than whichever one is installed -- see splet/alignment.py.
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
    """Pin the disagreement as a documented fact.

    Otherwise it is a surprise found later in a results table. If a future
    change makes the two backends break ties identically, this test fails and
    should be deleted -- together with the warning in splet/alignment.py and
    the reason the default backend is not "auto".
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


def test_resolve_backend_rejects_what_it_cannot_run(monkeypatch):
    import splet.alignment

    with pytest.raises(ValueError, match="unknown backend"):
        splet.alignment.resolve_backend("editdistance")
    monkeypatch.setattr(splet.alignment, "_RapidfuzzLevenshtein", None)
    with pytest.raises(ImportError, match="requires rapidfuzz"):
        splet.alignment.resolve_backend("rapidfuzz")
    assert splet.alignment.resolve_backend("auto") == "python"


def test_resolve_costs_accepts_a_name_or_a_model():
    import splet.alignment

    assert splet.alignment.resolve_costs("sclite") is COST_CHOICES["sclite"]
    model = CostModel(substitution=2.0, insertion=1.0, deletion=1.0)
    assert splet.alignment.resolve_costs(model) is model
    with pytest.raises(ValueError, match="unknown cost model"):
        splet.alignment.resolve_costs("sctk")


def test_sclite_costs_are_the_ones_sclite_documents():
    """Pin the weights against src/sclite/word.c:11-17.

    They are stated there as a comment and returned by wwd_WORD. Nothing
    derives them, so a typo here is silent until a corpus disagrees with
    sclite by a few tenths of a percent.
    """
    sclite = COST_CHOICES["sclite"]
    assert (sclite.deletion, sclite.insertion) == (3.0, 3.0)
    assert sclite.substitution == 4.0
    assert sclite.correct == 0.0
    assert sclite.optional_deletion == 2.0


def test_rapidfuzz_rejects_a_weighted_cost_model():
    pytest.importorskip("rapidfuzz")
    with pytest.raises(ValueError, match="cannot express a weighted cost model"):
        levenshtein_alignment(["a"], ["b"], backend="rapidfuzz", costs="sclite")


def test_sclite_costs_can_report_more_errors_than_unit_costs():
    """sclite minimises cost, not edit count, so it can report more errors.

    Four deletions and three insertions cost 21; the five substitutions and
    one deletion that give the minimum edit distance cost 23. Verified
    against sclite 2.10, which scores this pair 4 correct, 0 S, 4 D, 3 I.
    """
    ref = "b b b b a a a a".split()
    hyp = "a a a b a b b".split()

    sclite = levenshtein_alignment(ref, hyp, costs="sclite")
    assert (sclite.substitutions, sclite.deletions, sclite.insertions) == (0, 4, 3)
    assert (sclite.hits, sclite.errors) == (4, 7)

    unit = levenshtein_alignment(ref, hyp, costs="unit")
    assert unit.errors == 6


def test_sclite_costs_never_report_fewer_errors_than_unit_costs():
    """The minimum edit distance is a lower bound on what sclite reports."""
    import random

    random.seed(3)
    vocabulary = "a b c".split()
    for _ in range(300):
        ref = [random.choice(vocabulary) for _ in range(random.randint(0, 10))]
        hyp = [random.choice(vocabulary) for _ in range(random.randint(0, 10))]
        sclite = levenshtein_alignment(ref, hyp, costs="sclite")
        unit = levenshtein_alignment(ref, hyp, costs="unit")
        assert sclite.errors >= unit.errors, f"ref={ref} hyp={hyp}"


@pytest.mark.parametrize(
    "optional_deletion_is_correct, hits, substitutions, deletions",
    [(False, 1, 1, 1), (True, 2, 1, 0)],
)
def test_optional_reference_token(
    optional_deletion_is_correct, hits, substitutions, deletions
):
    """The example from doc/sclite.htm, scored both ways.

    sclite reports 1 C, 1 S, 1 D without -D and 2 C, 1 S, 0 D with it.
    """
    alignment = levenshtein_alignment(
        ["b", "(c)", "d"],
        ["b", "e"],
        optional_deletion_is_correct=optional_deletion_is_correct,
    )
    assert alignment.hits == hits
    assert alignment.substitutions == substitutions
    assert alignment.deletions == deletions
    assert alignment.insertions == 0


def test_an_optional_hypothesis_token_may_be_inserted_for_free():
    """sclite's -D covers inserted optional hypothesis words too.

    doc/sclite.htm: "optionality processing for hypothesis words" was added
    after the reference-side support the name still refers to.
    """
    ref = ["b", "d"]
    hyp = ["b", "(x)", "d"]
    assert levenshtein_alignment(ref, hyp).insertions == 1
    scored = levenshtein_alignment(ref, hyp, optional_deletion_is_correct=True)
    assert (scored.insertions, scored.hits) == (0, 3)


def test_optional_marking_is_stripped_only_for_the_comparison():
    """A marked token still prints with its parentheses.

    The alignment is rendered into reports and read beside the ref file, so
    a token that is not the one in that file is a bug report waiting to be
    filed.
    """
    alignment = levenshtein_alignment(["(c)"], ["c"])
    assert alignment.hits == 1
    assert alignment.operations == [("hit", "(c)", "c")]


def test_parentheses_alone_are_not_a_marking():
    """Empty parentheses are a two-character token, not an optional one."""
    alignment = levenshtein_alignment(["()"], [])
    assert alignment.deletions == 1
