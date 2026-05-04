import pytest

from espnet3.systems.base.inference import (
    _collect_scp_lines,
    _flatten_results,
    _materialize_output_value,
)


def test_flatten_results_preserves_order():
    assert _flatten_results([1, [2, 3], 4]) == [1, 2, 3, 4]


def test_collect_scp_lines_with_scalar_outputs():
    results = [
        {"idx": 0, "hyp": "h0", "ref": "r0"},
        {"idx": 1, "hyp": "h1", "ref": "r1"},
    ]
    scp_lines = _collect_scp_lines(
        results, idx_key="idx", hyp_keys="hyp", ref_keys="ref"
    )
    assert scp_lines["hyp"] == ["0 h0", "1 h1"]
    assert scp_lines["ref"] == ["0 r0", "1 r1"]


def test_materialize_output_value_rejects_top_level_lists(tmp_path):
    with pytest.raises(TypeError, match="Top-level list outputs are not supported"):
        _materialize_output_value(
            idx_value="utt1",
            field_key="hyp",
            value=["h1", "h2"],
            output_dir=tmp_path,
            artifact_config=None,
        )
