"""Tests for the speechlm/parallel_utils/__init__.py registry."""

import pytest

pytest.importorskip("torchtitan", reason="torchtitan not installed")

from espnet2.speechlm.model.speechlm.parallel_utils import (  # noqa: E402
    parallel_strategies,
)


class TestRegistry:
    def test_qwen3_registered(self):
        assert "qwen3" in parallel_strategies

    def test_registered_is_callable(self):
        assert callable(parallel_strategies["qwen3"])
