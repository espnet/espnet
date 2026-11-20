"""Tests of the label check's scoring (align_label_check.py score)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_label_check import Scorer, grid_search, labels, merge, widen  # noqa: E402


def test_labels_pad_start_and_end_separately():
    rows = [("r", "u", 1.0, 1.2), ("r", "u", 1.3, 1.5), ("r", "v", 4.0, 4.5)]
    # the 0.1 s pause inside u is bridged at min_pause 0.2, kept at 0.05
    assert labels(rows, 0.2, 0.0, 0.0) == {"r": [(1.0, 1.5), (4.0, 4.5)]}
    assert labels(rows, 0.05, 0.0, 0.0) == {"r": [(1.0, 1.2), (1.3, 1.5), (4.0, 4.5)]}
    assert labels(rows, 0.2, 0.1, 0.3) == {"r": [(0.9, 1.8), (3.9, 4.8)]}


def test_scorer_counts_miss_and_false_alarm_in_frames():
    scorer = Scorer({"r": [(1.0, 2.0)]}, {"r": 5.0})
    ev = scorer({"r": [(1.5, 2.5)]})
    assert ev["miss"] == pytest.approx(50.0)
    assert ev["fa"] == pytest.approx(50.0)
    assert ev["deter"] == pytest.approx(100.0)


def test_grid_search_finds_the_padding_that_matches_the_reference():
    ref = {"r": merge([(1.0, 2.0), (3.0, 4.0)])}
    scorer = Scorer(ref, {"r": 6.0})
    # labels start 50 ms late and end 300 ms early
    rows = [("r", "u", 1.05, 1.7), ("r", "v", 3.05, 3.7)]
    best = grid_search(rows, scorer, [0, 0.05], [0, 0.1, 0.3, 0.5], [0.2])[0]
    assert best[0] == pytest.approx(0.0)
    assert (best[3], best[4]) == (0.05, 0.3)


def test_widen_extends_only_a_dimension_whose_best_is_its_largest_value():
    assert widen([0, 0.1, 0.2], 0.2, 0.1) == [0, 0.1, 0.2, 0.3, 0.4]
    assert widen([0, 0.1, 0.2], 0.1, 0.1) == [0, 0.1, 0.2]


def test_pauses_split_inside_and_between_utterances():
    from align_label_check import labelled_segments, split_pauses

    rows = [
        ("r", "u", 1.0, 1.2),
        ("r", "u", 2.0, 2.2),  # 0.8 s inside u: kept at min_pause 0.5
        ("r", "v", 5.0, 5.5),  # 2.8 s between u and v
    ]
    inside, between = split_pauses(labelled_segments(rows, 0.5, 0.0, 0.0))
    assert inside == pytest.approx([0.8])
    assert between == pytest.approx([2.8])
    inside, between = split_pauses(labelled_segments(rows, float("inf"), 0.0, 0.0))
    assert inside == [] and between == pytest.approx([2.8])
