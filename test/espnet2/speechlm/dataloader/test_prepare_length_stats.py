"""Tests for espnet2/speechlm/bin/prepare_length_stats.py worker function."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub espnet2.speechlm.model to avoid transitive transformers import
# (which fails when pyarrow is stubbed by conftest.py without __version__).
if "espnet2.speechlm.model" not in sys.modules:
    _stub = types.ModuleType("espnet2.speechlm.model")
    _stub._all_job_types = {}
    sys.modules["espnet2.speechlm.model"] = _stub

from espnet2.speechlm.bin.prepare_length_stats import worker  # noqa: E402


class TestWorker:
    def test_returns_empty_on_value_error(self):
        preprocessor = MagicMock()
        with patch(
            "espnet2.speechlm.bin.prepare_length_stats.DataIteratorFactory"
        ) as MockFactory:
            MockFactory.return_value.build_iter.side_effect = ValueError("bad shard")
            result = worker(
                preprocessor=preprocessor,
                rank=0,
                world_size=1,
                unregistered_spec="asr:dummy:path.json",
            )
        assert result == {}
        preprocessor.find_length.assert_not_called()

    def test_propagates_non_value_error(self):
        preprocessor = MagicMock()
        with patch(
            "espnet2.speechlm.bin.prepare_length_stats.DataIteratorFactory"
        ) as MockFactory:
            MockFactory.return_value.build_iter.side_effect = RuntimeError("fatal")
            with pytest.raises(RuntimeError, match="fatal"):
                worker(
                    preprocessor=preprocessor,
                    rank=0,
                    world_size=1,
                    unregistered_spec="asr:dummy:path.json",
                )
