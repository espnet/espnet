"""Tests for espnet2/speechlm/bin/prepare_length_stats.py worker function."""

from unittest.mock import MagicMock, patch

import pytest

from espnet2.speechlm.bin.prepare_length_stats import worker


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
