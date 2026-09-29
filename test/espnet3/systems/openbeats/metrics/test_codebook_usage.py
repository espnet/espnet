"""Tests for the OpenBEATs codebook usage metric."""

import math

import pytest

from espnet3.systems.openbeats.metrics.codebook_usage import CodebookUsage

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                  | Description                   |
# |--------------------------------------------|-------------------------------|
# | test_codebook_usage_summarizes_targets     | Usage, entropy, perplexity.   |
# | test_uniform_codes_reach_codebook_size     | Uniform -> perplexity = K.    |
# | test_collapsed_codes_have_perplexity_one   | One code -> perplexity 1.     |
# | test_out_of_range_ids_raise                | id >= codebook_size raises.   |
# | test_empty_targets_raise                   | No ids raises.                |
# | test_invalid_codebook_size_raises          | codebook_size <= 0 raises.    |


def _scp(tmp_path, text):
    path = tmp_path / "target.scp"
    path.write_text(text, encoding="utf-8")
    return {"target": path}


def test_codebook_usage_summarizes_targets(tmp_path):
    data = _scp(tmp_path, "0 3 3 1 2\n1 3 0\n")

    result = CodebookUsage(codebook_size=4)(data, "valid", tmp_path)

    assert result == {
        "num_tokens": 6,
        "used_codes": 4,
        "usage": 1.0,
        "entropy": 1.7925,
        "perplexity": 3.4641,
    }


def test_uniform_codes_reach_codebook_size(tmp_path):
    data = _scp(tmp_path, "0 0 1 2 3\n1 3 2 1 0\n")

    result = CodebookUsage(codebook_size=8)(data, "valid", tmp_path)

    assert result["usage"] == 0.5
    assert math.isclose(result["perplexity"], 4.0)


def test_collapsed_codes_have_perplexity_one(tmp_path):
    data = _scp(tmp_path, "0 5 5 5\n1 5\n")

    result = CodebookUsage(codebook_size=16)(data, "valid", tmp_path)

    assert result["used_codes"] == 1
    assert result["entropy"] == 0.0
    assert result["perplexity"] == 1.0


def test_out_of_range_ids_raise(tmp_path):
    data = _scp(tmp_path, "0 1 16\n")

    with pytest.raises(ValueError, match="codebook_size"):
        CodebookUsage(codebook_size=16)(data, "valid", tmp_path)


def test_empty_targets_raise(tmp_path):
    data = _scp(tmp_path, "0\n")

    with pytest.raises(ValueError, match="no target ids"):
        CodebookUsage(codebook_size=16)(data, "valid", tmp_path)


def test_invalid_codebook_size_raises():
    with pytest.raises(ValueError, match="positive"):
        CodebookUsage(codebook_size=0)
