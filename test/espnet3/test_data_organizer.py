# test_data_organizer.py
import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet2.train.preprocessor import AbsPreprocessor
from espnet3.data import CombinedDataset, DataOrganizer, do_nothing_transform

# ===============================================================
# Test Case Summary for DataOrganizer & CombinedDataset
# ===============================================================
#
# Basic Integration
# | Test Function Name               | Description                                                              | # noqa: E501
# |----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_data_organizer_init         | Initializes DataOrganizer with train/valid/test                          | # noqa: E501
# | test_data_organizer_without_test | Verifies behavior without test section                                   | # noqa: E501
# | test_data_organizer_test_only    | Validates usage of test-only pipelines                                   | # noqa: E501
# | test_data_organizer_train_valid_multiple | Ensures multiple train/valid datasets are supported             | # noqa: E501
# | test_data_organizer_empty_train_valid_ok | Confirms train/valid can be empty lists                          | # noqa: E501
#
# Transform / Preprocessor Application
# | Test Function Name                    | Description                                                              | # noqa: E501
# |--------------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_combined_dataset                | Combines datasets and applies transform                                  | # noqa: E501
# | test_data_organizer_transform_only   | Applies only transform to data                                           | # noqa: E501
# | test_data_organizer_preprocessor_only| Applies only preprocessor to data                                        | # noqa: E501
# | test_data_organizer_transform_and_preprocessor | Applies both transform and preprocessor                        | # noqa: E501
# | test_espnet_preprocessor_without_transform | Uses only ESPnet-style preprocessor (UID-based)               | # noqa: E501
# | test_espnet_preprocessor_with_transform    | Combines transform with ESPnet preprocessor                   | # noqa: E501
#
# Test Set Variants
# | Test Function Name                 | Description                                                              | # noqa: E501
# |-----------------------------------|--------------------------------------------------------------------------| # noqa: E501
# | test_data_organizer_test_multiple_sets | Handles multiple named test sets                                   | # noqa: E501
#
# Error Cases
# | Test Function Name                         | Description                                                  | Expected Exception | # noqa: E501
# |--------------------------------------------|--------------------------------------------------------------|--------------------| # noqa: E501
# | test_data_organizer_train_only_assertion   | Raises error when only train is provided without valid       | RuntimeError       | # noqa: E501
# | test_data_organizer_inconsistent_keys      | Fails when dataset output keys are inconsistent in combined  | AssertionError     | # noqa: E501
# | test_data_organizer_transform_none         | Simulates transform failure that raises an internal exception| ValueError         | # noqa: E501
# | test_data_organizer_invalid_preprocessor_type | Fails when a non-callable is used as preprocessor         | AssertionError     | # noqa: E501


# Dummy classes
class DummyTransform:
    def __call__(self, sample):
        return {
            "audio": sample["audio"],
            "text": sample["text"].upper(),
        }


class DummyPreprocessor:
    def __call__(self, sample):
        return {
            "audio": sample["audio"],
            "text": f"[dummy] {sample['text']}",
        }


class ESPnetPreprocessor(AbsPreprocessor):
    def __init__(self):
        super().__init__(train=True)

    def __call__(self, uid, sample):
        return {
            "audio": sample["audio"],
            "text": f"[espnet] {sample['text']}",
        }


class DummyDataset:
    def __init__(self, path=None):
        self.data = [
            {"audio": np.random.random(16000), "text": "hello"},
            {"audio": np.random.random(16000), "text": "world"},
        ]

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
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                "transform": {
                    "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                },
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
        "test": [
            {
                "name": "test_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
    }
    config = OmegaConf.create(config)
    return config


# Tests
def test_combined_dataset():
    ds1 = DummyDataset()
    ds2 = DummyDataset()
    combined = CombinedDataset(
        [ds1, ds2],
        [
            (DummyTransform(), do_nothing_transform),
            (DummyTransform(), do_nothing_transform),
        ],
    )
    assert len(combined) == 4
    assert combined[0]["text"] == "HELLO"
    assert combined[3]["text"] == "WORLD"


def test_data_organizer_init(dummy_dataset_config):
    config = dummy_dataset_config
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        test=instantiate(config["test"]),
        preprocessor=DummyPreprocessor(),
    )
    assert len(organizer.train) == 2
    assert organizer.train[0]["text"] == "[dummy] HELLO"
    assert len(organizer.valid) == 2
    assert organizer.valid[0]["text"] == "[dummy] hello"
    assert "test_dummy" in organizer.test


def test_data_organizer_without_test():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                "transform": {
                    "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                },
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
        # No "test" field
    }
    config = OmegaConf.create(config)
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=DummyPreprocessor(),
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
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                "transform": {
                    "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                },
            }
        ],
    }

    organizer = DataOrganizer(
        test=instantiate(config["test"]), preprocessor=DummyPreprocessor()
    )

    assert organizer.train is None
    assert organizer.valid is None
    assert "test_dummy" in organizer.test
    assert organizer.test["test_dummy"][0]["text"] == "[dummy] HELLO"


@pytest.mark.parametrize("train_count, valid_count", [(1, 1), (2, 2)])
def test_data_organizer_train_valid_multiple(train_count, valid_count):
    config = {
        "train": [
            {
                "name": f"train{i}",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                "transform": {
                    "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                },
            }
            for i in range(train_count)
        ],
        "valid": [
            {
                "name": f"valid{i}",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
            for i in range(valid_count)
        ],
    }
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=DummyPreprocessor(),
    )
    assert len(organizer.train) == 2 * train_count
    assert len(organizer.valid) == 2 * valid_count
    assert organizer.train[0]["text"] == "[dummy] HELLO"
    assert organizer.valid[0]["text"] == "[dummy] hello"


def test_data_organizer_test_multiple_sets():
    config = {
        "test": [
            {
                "name": "test_clean",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            },
            {
                "name": "test_other",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                "transform": {
                    "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                },
            },
        ]
    }
    organizer = DataOrganizer(
        test=instantiate(config["test"]), preprocessor=DummyPreprocessor()
    )
    assert "test_clean" in organizer.test
    assert "test_other" in organizer.test


def test_data_organizer_transform_only():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                "transform": {
                    "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                },
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
    }
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
    )
    assert organizer.train[0]["text"] == "HELLO"
    assert organizer.valid[0]["text"] == "hello"


def test_data_organizer_preprocessor_only():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
    }
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=DummyPreprocessor(),
    )
    assert organizer.train[0]["text"] == "[dummy] hello"
    assert organizer.valid[0]["text"] == "[dummy] hello"


def test_data_organizer_transform_and_preprocessor():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                "transform": {
                    "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                },
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
    }
    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=DummyPreprocessor(),
    )
    assert organizer.train[0]["text"] == "[dummy] HELLO"
    assert organizer.valid[0]["text"] == "[dummy] hello"


def test_data_organizer_train_only_assertion():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
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
        CombinedDataset(
            [ds1, ds2],
            [
                (do_nothing_transform, do_nothing_transform),
                (do_nothing_transform, do_nothing_transform),
            ],
        )


def test_data_organizer_transform_none():
    class BrokenTransform:
        def __call__(self, sample):
            raise ValueError("Broken")

    ds = DummyDataset()
    with pytest.raises(ValueError):
        CombinedDataset([ds], [(BrokenTransform(), do_nothing_transform)])


def test_data_organizer_invalid_preprocessor_type():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
    }
    with pytest.raises(AssertionError):
        DataOrganizer(
            train=instantiate(config["train"]),
            valid=instantiate(config["valid"]),
            preprocessor="not_callable",
        )


def test_espnet_preprocessor_without_transform():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                # transform is omitted
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
    }

    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=ESPnetPreprocessor(),  # ESPnet-style preprocessor
    )

    sample = organizer.train[0]
    assert sample["text"] == "[espnet] hello"


def test_espnet_preprocessor_with_transform():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
                "transform": {
                    "_target_": "test.espnet3.test_data_organizer.DummyTransform"
                },
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": "test.espnet3.test_data_organizer.DummyDataset"
                },
            }
        ],
    }

    organizer = DataOrganizer(
        train=instantiate(config["train"]),
        valid=instantiate(config["valid"]),
        preprocessor=ESPnetPreprocessor(),  # ESPnet-style preprocessor
    )

    sample = organizer.train[0]
    assert sample["text"] == "[espnet] HELLO"
