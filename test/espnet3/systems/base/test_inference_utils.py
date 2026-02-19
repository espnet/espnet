import pytest

from espnet3.systems.base.inference import _collect_scp_lines, _flatten_results


def test_flatten_results_preserves_order():
    assert _flatten_results([1, [2, 3], 4]) == [1, 2, 3, 4]


def test_collect_scp_lines_with_list_outputs():
    results = [
        {"idx": 0, "hyp": ["h1", "h2"], "ref": "r0"},
        {"idx": 1, "hyp": ["h3", "h4"], "ref": "r1"},
    ]
    scp_lines = _collect_scp_lines(
        results, idx_key="idx", hyp_keys="hyp", ref_keys="ref"
    )
    assert scp_lines["hyp0"] == ["0 h1", "1 h3"]
    assert scp_lines["hyp1"] == ["0 h2", "1 h4"]
    assert scp_lines["ref"] == ["0 r0", "1 r1"]


def test_collect_scp_lines_rejects_mismatched_lists():
    results = [
        {"idx": 0, "hyp": ["h1", "h2"], "ref": "r0"},
        {"idx": 1, "hyp": ["h3"], "ref": "r1"},
    ]
    with pytest.raises(ValueError):
        _collect_scp_lines(results, idx_key="idx", hyp_keys="hyp", ref_keys="ref")
