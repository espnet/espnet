"""Tests for espnet2/speechlm/bin/prepare_length_stats.py worker function."""

import json
from unittest.mock import MagicMock, patch

import pytest

from espnet2.speechlm.bin.prepare_length_stats import worker


def test_main_collects_real_data_from_quoted_path(tmp_path, monkeypatch):
    from espnet2.speechlm.bin import prepare_length_stats as module

    directory = tmp_path / "space dir"
    directory.mkdir()
    text = directory / "text.txt"
    text.write_text("utt1 hello world\n")
    manifest = directory / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "data_entry": [{"name": "text1", "path": str(text), "reader": "text"}],
                "samples": ["utt1"],
            }
        )
    )
    config = tmp_path / "config.yaml"
    config.write_text("job_type: test\n")
    output = tmp_path / "stats"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_length_stats",
            "--train-config",
            str(config),
            "--output-dir",
            str(output),
            "--train-unregistered-specifier",
            f"text_only:train:'{manifest}'",
            "--valid-unregistered-specifier",
            f"'text_only:valid:{manifest}'",
        ],
    )
    preprocessor = MagicMock()
    preprocessor.find_length.return_value = 7
    job = MagicMock()
    job.build_preprocessor.return_value = preprocessor

    def collect(preprocessor, num_workers, spec_type, specifier):
        return worker(preprocessor, 0, 1, unregistered_spec=specifier)

    with (
        patch.object(module, "_all_job_types", {"test": lambda *a, **k: job}),
        patch.object(module, "collect_length_stats", side_effect=collect),
    ):
        module.main()
    assert preprocessor.find_length.call_count == 2
    for split in ["train", "valid"]:
        assert json.loads((output / f"stats_text_only_{split}.jsonl").read_text()) == {
            "utt1": 7
        }


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
