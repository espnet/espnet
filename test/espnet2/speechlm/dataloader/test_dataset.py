"""Tests for espnet2/speechlm/dataloader/dataset.py."""

import json
import os
from unittest.mock import patch

import pytest
import yaml

# ---------- MockReader for dataset tests ----------


class MockReader:
    """Simple dict-like reader that reads a JSON file {id: value}."""

    def __init__(self, path, valid_ids=None):
        with open(path, "r") as f:
            data = json.load(f)
        if valid_ids is not None:
            valid_set = set(valid_ids)
            self.data = {k: v for k, v in data.items() if k in valid_set}
        else:
            self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()


MOCK_LOADERS = {"mock_reader": MockReader}


def _make_dataset_json(tmp_path, name, samples, reader_type="mock_reader"):
    """Helper: create a dataset JSON + mock data file."""
    data_file = tmp_path / f"{name}_data.json"
    data_file.write_text(json.dumps({s: f"value_{s}" for s in samples}))

    dataset_json = tmp_path / f"{name}.json"
    dataset_json.write_text(
        json.dumps(
            {
                "data_entry": [
                    {
                        "name": "text1",
                        "path": str(data_file),
                        "reader": reader_type,
                    }
                ],
                "samples": samples,
            }
        )
    )
    return str(dataset_json)


# ---------- SingleDataset ----------


class TestSingleDataset:
    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_init_loads_json(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import SingleDataset

        json_path = _make_dataset_json(tmp_path, "ds1", ["utt1", "utt2", "utt3"])
        ds = SingleDataset(json_path)
        assert len(ds) == 3
        assert "text1" in ds.entries
        assert set(ds.sample_ids) == {"utt1", "utt2", "utt3"}

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_getitem(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import SingleDataset

        json_path = _make_dataset_json(tmp_path, "ds1", ["utt1", "utt2"])
        ds = SingleDataset(json_path)
        item = ds["utt1"]
        assert item == {"text1": "value_utt1"}

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_distributed_sharding(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import SingleDataset

        samples = ["utt0", "utt1", "utt2", "utt3"]
        json_path = _make_dataset_json(tmp_path, "ds1", samples)
        ds0 = SingleDataset(json_path, rank=0, world_size=2)
        ds1 = SingleDataset(json_path, rank=1, world_size=2)
        assert set(ds0.sample_ids) == {"utt0", "utt2"}
        assert set(ds1.sample_ids) == {"utt1", "utt3"}

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_unknown_reader_raises(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import SingleDataset

        json_path = _make_dataset_json(
            tmp_path, "ds1", ["utt1"], reader_type="nonexistent_reader"
        )
        with pytest.raises(ValueError, match="Unknown reader type"):
            SingleDataset(json_path)


# ---------- CombinedDataset ----------


class TestCombinedDataset:
    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_from_direct_paths(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import CombinedDataset

        path1 = _make_dataset_json(tmp_path, "ds1", ["a", "b"])
        path2 = _make_dataset_json(tmp_path, "ds2", ["c", "d"])
        combined = CombinedDataset(
            datasets=[("ds1", path1), ("ds2", path2)], num_worker=1
        )
        assert set(combined.dataset_names) == {"ds1", "ds2"}
        assert len(combined) == 4

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_duplicate_name_raises(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import CombinedDataset

        path1 = _make_dataset_json(tmp_path, "ds1", ["a"])
        with pytest.raises(ValueError, match="Duplicate dataset name"):
            CombinedDataset(
                datasets=[("samename", path1), ("samename", path1)], num_worker=1
            )

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_getitem(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import CombinedDataset

        path1 = _make_dataset_json(tmp_path, "ds1", ["utt1"])
        combined = CombinedDataset(datasets=[("ds1", path1)], num_worker=1)
        key, data = combined[("task", "ds1", "utt1")]
        assert key == ("task", "ds1", "utt1")
        assert data == {"text1": "value_utt1"}

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_verify_entries_pass(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import CombinedDataset

        path1 = _make_dataset_json(tmp_path, "ds1", ["a"])
        combined = CombinedDataset(datasets=[("ds1", path1)], num_worker=1)
        # "text1" exists in the dataset entries, so this should not raise
        combined.verify_subset_entries("asr", "ds1", ["text1"])

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_verify_entries_fail(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import CombinedDataset

        path1 = _make_dataset_json(tmp_path, "ds1", ["a"])
        combined = CombinedDataset(datasets=[("ds1", path1)], num_worker=1)
        with pytest.raises(ValueError, match="requires entry"):
            combined.verify_subset_entries("asr", "ds1", ["nonexistent_entry"])

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_get_all_examples(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import CombinedDataset

        path1 = _make_dataset_json(tmp_path, "ds1", ["a", "b"])
        path2 = _make_dataset_json(tmp_path, "ds2", ["c"])
        combined = CombinedDataset(
            datasets=[("ds1", path1), ("ds2", path2)], num_worker=1
        )
        examples = combined.get_all_examples()
        assert set(examples["ds1"]) == {"a", "b"}
        assert set(examples["ds2"]) == {"c"}

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_registry_loading(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import CombinedDataset

        # Create a dataset JSON and a registry YAML pointing to it
        path1 = _make_dataset_json(tmp_path, "ds1", ["a", "b"])
        registry = tmp_path / "registry.yaml"
        registry.write_text(yaml.dump({"reg_ds": {"path": path1}}))

        with patch.dict(os.environ, {"ESPNET_DATASET_REGISTRY": str(registry)}):
            combined = CombinedDataset(registered_datasets=["reg_ds"], num_worker=1)
        assert "reg_ds" in combined.dataset_names

    @patch(
        "espnet2.speechlm.dataloader.dataset.ALL_DATA_LOADERS",
        MOCK_LOADERS,
    )
    def test_registered_not_found(self, tmp_path):
        from espnet2.speechlm.dataloader.dataset import CombinedDataset

        registry = tmp_path / "registry.yaml"
        registry.write_text(yaml.dump({"existing_ds": {"path": "/fake"}}))

        with patch.dict(os.environ, {"ESPNET_DATASET_REGISTRY": str(registry)}):
            with pytest.raises(ValueError, match="not found in registry"):
                CombinedDataset(registered_datasets=["nonexistent_ds"], num_worker=1)
