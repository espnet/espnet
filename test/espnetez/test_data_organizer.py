from types import SimpleNamespace

import pytest

from espnet3.data import CombinedDataset, DataOrganizer, DatasetConfig


# Dummy transform and preprocessor classes for testing
class DummyTransform:
    def __call__(self, sample):
        sample["text"] = sample["text"].upper()
        return sample


class DummyPreprocessor:
    def __call__(self, sample):
        sample["text"] = f"[DUMMY] {sample['text']}"
        return sample


# Dummy dataset for testing
class DummyDataset:
    def __init__(self, path=None):
        self.data = [
            {"audio": b"audio1", "text": "hello"},
            {"audio": b"audio2", "text": "world"},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EmptyDataset:
    def __init__(self, path=None):
        self.data = []

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("Empty dataset")


@pytest.fixture
def dummy_dataset_config():
    return {
        "train": [
            {
                "name": "dummy",
                "dataset_cls": "test.espnetez.test_data_organizer.DummyDataset",
                "fields": ["audio", "text"],
                "transform": "test.espnetez.test_data_organizer.DummyTransform",
            }
        ],
        "valid": [
            {
                "name": "dummy",
                "dataset_cls": "test.espnetez.test_data_organizer.DummyDataset",
                "fields": ["audio", "text"],
            }
        ],
        "test": {
            "sample_test": {
                "name": "dummy",
                "dataset_cls": "test.espnetez.test_data_organizer.DummyDataset",
                "path": None,
                "fields": ["audio", "text"],
            }
        },
    }


def test_combined_dataset():
    ds1 = DummyDataset()
    ds2 = DummyDataset()
    combined = CombinedDataset([ds1, ds2], [DummyTransform(), DummyTransform()])
    assert len(combined) == 4
    assert combined[0]["text"] == "HELLO"
    assert combined[3]["text"] == "WORLD"


def test_combined_dataset_index_error():
    ds = DummyDataset()
    combined = CombinedDataset([ds], [DummyTransform()])
    with pytest.raises(IndexError):
        _ = combined[10]


def test_empty_combined_dataset():
    empty = CombinedDataset([], [])
    assert len(empty) == 0
    with pytest.raises(IndexError):
        _ = empty[0]


def test_data_organizer_init(dummy_dataset_config):
    organizer = DataOrganizer(dummy_dataset_config, preprocessor=DummyPreprocessor())

    # train set
    assert len(organizer.train) == 2
    assert organizer.train[0]["text"].startswith("[DUMMY]")

    # valid set
    assert len(organizer.valid) == 2

    # test set
    assert "sample_test" in organizer.test
    assert len(organizer.test["sample_test"]) == 2
    assert isinstance(organizer.test["sample_test"][0], dict)


def test_data_organizer_empty_dataset():
    config = {
        "train": [
            {
                "name": "empty",
                "dataset_cls": "test.espnetez.test_data_organizer.EmptyDataset",
                "fields": ["audio", "text"],
            }
        ],
        "valid": [
            {
                "name": "empty",
                "dataset_cls": "test.espnetez.test_data_organizer.EmptyDataset",
                "fields": ["audio", "text"],
            }
        ],
    }
    organizer = DataOrganizer(config)
    assert len(organizer.train) == 0
    assert len(organizer.valid) == 0


def test_missing_train_valid_raises():
    incomplete_cfg = {"train": []}  # missing valid
    with pytest.raises(AssertionError):
        DataOrganizer(incomplete_cfg)


def test_dataset_config_from_dict():
    cfg_dict = {
        "name": "dummy",
        "dataset_cls": "test.espnetez.test_data_organizer.DummyDataset",
        "fields": ["audio", "text"],
        "transform": "test.espnetez.test_data_organizer.DummyTransform",
    }
    cfg = DatasetConfig.from_dict(cfg_dict)
    assert cfg.name == "dummy"
    assert callable(cfg.transform)
