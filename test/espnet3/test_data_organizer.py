# test_data_organizer.py
import pytest
from espnet3.data import CombinedDataset, DataOrganizer
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Dummy classes
class DummyTransform:
    def __call__(self, sample):
        sample["text"] = sample["text"].upper()
        return sample

class DummyPreprocessor:
    def __call__(self, sample):
        sample["text"] = f"[DUMMY] {sample['text']}"
        return sample

class DummyDataset:
    def __init__(self, path=None):
        self.data = [{"audio": b"audio1", "text": "hello"}, {"audio": b"audio2", "text": "world"}]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Fixtures
@pytest.fixture
def dummy_dataset_config():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
                "transform": {"_target_": "test.espnet3.test_data_organizer.DummyTransform"},
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ],
        "test": [
            {
                "name": "test_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ],
    }
    config = OmegaConf.create(config)
    return config

# Tests
def test_combined_dataset():
    ds1 = DummyDataset()
    ds2 = DummyDataset()
    combined = CombinedDataset([ds1, ds2], [DummyTransform(), DummyTransform()])
    assert len(combined) == 4
    assert combined[0]["text"] == "HELLO"
    assert combined[3]["text"] == "WORLD"

def test_data_organizer_init(dummy_dataset_config):
    config = dummy_dataset_config
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        test=instantiate(config["test"]),
        preprocessor=DummyPreprocessor()
    )

    assert len(organizer.train) == 2
    assert organizer.train[0]["text"].startswith("[DUMMY]")
    assert len(organizer.valid) == 2
    assert "test_dummy" in organizer.test
    assert isinstance(organizer.test["test_dummy"][0], tuple)

def test_data_organizer_without_test():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
                "transform": {"_target_": "test.espnet3.test_data_organizer.DummyTransform"},
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ],
        # No "test" field
    }
    config = OmegaConf.create(config)
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=DummyPreprocessor()
    )

    assert len(organizer.train) == 2
    assert len(organizer.valid) == 2
    assert organizer.test == {}

def test_data_organizer_test_only():
    config = {
        # No "train"
        # No "valid"
        "test": [
            {
                "name": "test_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
                "transform": {"_target_": "test.espnet3.test_data_organizer.DummyTransform"},
            }
        ],
    }

    organizer = DataOrganizer(
        test=instantiate(config["test"]),
        preprocessor=DummyPreprocessor()
    )

    assert organizer.train is None
    assert organizer.valid is None
    assert "test_dummy" in organizer.test
    print(organizer.test["test_dummy"].dataset)
    assert organizer.test["test_dummy"][0][1]["text"].startswith("[DUMMY]")


@pytest.mark.parametrize("train_count, valid_count", [(1, 1), (2, 2)])
def test_data_organizer_train_valid_multiple(train_count, valid_count):
    config = {
        "train": [
            {
                "name": f"train{i}",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
                "transform": {"_target_": "test.espnet3.test_data_organizer.DummyTransform"},
            }
            for i in range(train_count)
        ],
        "valid": [
            {
                "name": f"valid{i}",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
            for i in range(valid_count)
        ]
    }
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=DummyPreprocessor()
    )
    assert len(organizer.train) == 2 * train_count
    assert len(organizer.valid) == 2 * valid_count
    assert organizer.train[0]["text"].startswith("[DUMMY]")


def test_data_organizer_test_multiple_sets():
    config = {
        "test": [
            {
                "name": "test_clean",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            },
            {
                "name": "test_other",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
                "transform": {"_target_": "test.espnet3.test_data_organizer.DummyTransform"},
            }
        ]
    }
    organizer = DataOrganizer(
        test=instantiate(config["test"]),
        preprocessor=DummyPreprocessor()
    )
    assert "test_clean" in organizer.test
    assert "test_other" in organizer.test


def test_data_organizer_transform_only():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
                "transform": {"_target_": "test.espnet3.test_data_organizer.DummyTransform"},
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ],
    }
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
    )
    assert organizer.train[0]["text"] == "HELLO"


def test_data_organizer_preprocessor_only():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ],
    }
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=DummyPreprocessor()
    )
    assert organizer.train[0]["text"].startswith("[DUMMY]")


def test_data_organizer_transform_and_preprocessor():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
                "transform": {"_target_": "test.espnet3.test_data_organizer.DummyTransform"},
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ],
    }
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=DummyPreprocessor()
    )
    sample = organizer.train[0]
    assert sample["text"] == "[DUMMY] [DUMMY] HELLO"


def test_data_organizer_train_only_assertion():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ]
        # valid is missing
    }
    with pytest.raises(RuntimeError):
        DataOrganizer(train=instantiate(config["train"]))


def test_data_organizer_empty_train_valid_ok():
    config = {
        "train": [],
        "valid": [],
    }
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
    )
    assert len(organizer.train) == 0
    assert len(organizer.valid) == 0


def test_data_organizer_inconsistent_keys():
    class BadDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"audio": b"abc", "wrong": "oops"} 

    ds1 = DummyDataset()
    ds2 = BadDataset()

    with pytest.raises(AssertionError):
        CombinedDataset([ds1, ds2], [lambda x: x, lambda x: x])


def test_data_organizer_transform_none():
    class BrokenTransform:
        def __call__(self, sample):
            raise ValueError("Broken")

    ds = DummyDataset()
    with pytest.raises(ValueError):
        CombinedDataset([ds], [BrokenTransform()])


def test_data_organizer_invalid_preprocessor_type():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {"_target_": "test.espnet3.test_data_organizer.DummyDataset"},
            }
        ],
    }
    with pytest.raises(TypeError):
        DataOrganizer(
            train=instantiate(config["train"]),
            valid=instantiate(config["valid"]),
            preprocessor="not_callable"
        )
