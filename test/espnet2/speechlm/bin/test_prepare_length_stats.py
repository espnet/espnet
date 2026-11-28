"""Tests for espnet2/speechlm/bin/prepare_length_stats.py worker function."""

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml

from espnet2.speechlm.bin.prepare_length_stats import save_stats, worker


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize(
    "directory_name", ["space dir", "reader's files", 'back\\slash "files"']
)
def test_main_collects_real_data_from_literal_arguments(
    tmp_path, monkeypatch, registered, directory_name
):
    from espnet2.speechlm.bin import prepare_length_stats as module

    directory = tmp_path / directory_name
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
    names = ["reader's train", 'back\\slash "train"', "valid"]
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump({name: {"path": str(manifest)} for name in names})
    )
    monkeypatch.setenv("ESPNET_DATASET_REGISTRY", str(registry))
    route = "registered" if registered else "unregistered"
    specifiers = [
        f"text_only:{name}" + ("" if registered else f":{manifest}") for name in names
    ]
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_length_stats",
            "--train-config",
            str(config),
            "--output-dir",
            str(output),
            f"--train-{route}-specifier",
            specifiers[0],
            f"--train-{route}-specifier",
            specifiers[1],
            f"--valid-{route}-specifier",
            specifiers[2],
        ],
    )
    preprocessor = MagicMock()
    preprocessor.find_length.return_value = 7
    job = MagicMock()
    job.build_preprocessor.return_value = preprocessor

    def collect(preprocessor, num_workers, spec_type, specifier):
        return worker(preprocessor, 0, 1, **{f"{spec_type}_spec": specifier})

    with (
        patch.object(module, "_all_job_types", {"test": lambda *a, **k: job}),
        patch.object(module, "collect_length_stats", side_effect=collect),
    ):
        module.main()
        module.main()
        assert job.build_preprocessor.call_count == 1
    assert preprocessor.find_length.call_count == 3
    for name in names:
        assert json.loads((output / f"stats_text_only_{name}.jsonl").read_text()) == {
            "utt1": 7
        }


class TestWorker:
    def test_skips_only_empty_shards(self):
        preprocessor = MagicMock()
        with patch(
            "espnet2.speechlm.bin.prepare_length_stats.DataIteratorFactory"
        ) as MockFactory:
            MockFactory.return_value.batched_examples = []
            result = worker(
                preprocessor=preprocessor,
                rank=0,
                world_size=1,
                unregistered_spec="asr:dummy:path.json",
            )
        assert result == {}
        preprocessor.find_length.assert_not_called()

    @pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
    def test_propagates_invalid_data(self, error_type):
        preprocessor = MagicMock()
        with patch(
            "espnet2.speechlm.bin.prepare_length_stats.DataIteratorFactory"
        ) as MockFactory:
            MockFactory.side_effect = error_type("fatal")
            with pytest.raises(error_type, match="fatal"):
                worker(
                    preprocessor=preprocessor,
                    rank=0,
                    world_size=1,
                    unregistered_spec="asr:dummy:path.json",
                )


def test_failed_write_does_not_publish_partial_stats(tmp_path):
    output = tmp_path / "stats.jsonl"
    with pytest.raises(ValueError, match="No examples"):
        save_stats({}, output)
    data = {("text_only", "train", "one"): 4, ("text_only", "train", "two"): object()}
    with pytest.raises(TypeError):
        save_stats(data, output)
    assert list(tmp_path.iterdir()) == []
    save_stats({("text_only", "train", "one"): 4}, output)
    assert json.loads(output.read_text()) == {"one": 4}
    with pytest.raises(TypeError):
        save_stats(data, output)
    assert json.loads(output.read_text()) == {"one": 4}
