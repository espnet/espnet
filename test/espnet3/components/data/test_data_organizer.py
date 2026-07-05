# test_data_organizer.py
import logging

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet2.train.preprocessor import AbsPreprocessor
from espnet3.components.data import data_organizer as data_organizer_module
from espnet3.components.data.data_organizer import (
    DataOrganizer,
    do_nothing,
)
from espnet3.components.data.dataset import (
    CombinedDataset,
    DatasetWithTransform,
    ShardedDataset,
)

# ===============================================================
# Test Case Summary for DataOrganizer & CombinedDataset
# ===============================================================

# | Test Function Name               | Description                              |
# |----------------------------------|------------------------------------------|
# | test_data_organizer_init         | Initializes DataOrganizer with           |
# |                                  | train/valid/test                         |
# | test_data_organizer_without_test | Verifies behavior without test section   |
# | test_data_organizer_test_only    | Validates usage of test-only pipelines   |
# | test_data_organizer_train_valid_multiple | Ensures multiple train/valid datasets   |
# |                                  | are supported                            |
# | test_data_organizer_empty_train_valid_ok | Confirms train/valid can be empty lists |
#
# Transform / Preprocessor Application
# | Test Function Name                    | Description                         |
# |--------------------------------------|--------------------------------------|
# | test_combined_dataset                | Combines datasets and applies transform  |
# | test_data_organizer_transform_only   | Applies only transform to data       |
# | test_data_organizer_preprocessor_only| Applies only preprocessor to data    |
# | test_data_organizer_transform_and_preprocessor | Applies both transform and |
# |                                      | preprocessor                         |
# | test_espnet_preprocessor_without_transform | Uses only ESPnet-style preprocessor   |
# |                                      | (UID-based)                          |
# | test_espnet_preprocessor_with_transform    | Combines transform with ESPnet |
# |                                      | preprocessor                         |
#
# Test Set Variants
# | Test Function Name                     | Description                               |
# |----------------------------------------|-------------------------------------------|
# | test_data_organizer_test_multiple_sets | Handles multiple named test sets          |
#
# Error Cases
# | Test Function Name                         | Description                           |
# |--------------------------------------------|---------------------------------------|
# | test_data_organizer_train_only_assertion   | Raises error when only train is       |
# |                                            | provided without valid                |
# | test_data_organizer_inconsistent_keys      | Fails when dataset output keys are    |
# |                                            | inconsistent in combined              |
# | test_data_organizer_transform_none         | Simulates transform failure that      |
# |                                            | raises an internal exception          |
# | test_data_organizer_invalid_preprocessor_type | Fails when a non-callable is used  |
# |                                            | as preprocessor                       |
# | test_combined_dataset_sharded_consistency_error | Ensures RuntimeError if only     |
# |                                            | some datasets use sharding            |
#
#  Expected Exceptions
# | Test Function Name                                 | Expected Exception |
# | -------------------------------------------------- | ------------------ |
# | test_data_organizer_train_only_assertion      | RuntimeError       |
# | test_data_organizer_inconsistent_keys          | AssertionError     |
# | test_data_organizer_transform_none             | ValueError         |
# | test_data_organizer_invalid_preprocessor_type | AssertionError     |
# | test_combined_dataset_sharded_consistency_error | RuntimeError       |

DUMMY_TRANSFORM_TARGET = (
    "test.espnet3.components.data.test_data_organizer.DummyTransform"
)
DUMMY_DATASET_TARGET = "test.espnet3.components.data.test_data_organizer.DummyDataset"
DUMMY_DATA_SRC = "dummy/asr"
DUMMY_SHARDED_DATA_SRC = "dummy/sharded"

DUMMY_PREPROCESSOR_TARGET = (
    "test.espnet3.components.data.test_data_organizer.DummyPreprocessor"
)
SPLIT_RECORDING_PREPROCESSOR_TARGET = (
    "test.espnet3.components.data.test_data_organizer.SplitRecordingPreprocessor"
)
ESPNET_TRAIN_FLAG_PREPROCESSOR_TARGET = (
    "test.espnet3.components.data." "test_data_organizer.TrainFlagRecordingPreprocessor"
)


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


class SplitRecordingPreprocessor:
    def __init__(self, tag: str):
        self.tag = tag

    def __call__(self, sample):
        return {
            "audio": sample["audio"],
            "text": f"[{self.tag}] {sample['text']}",
        }


class ESPnetPreprocessor(AbsPreprocessor):
    def __init__(self):
        super().__init__(train=True)

    def __call__(self, uid, sample):
        return {
            "audio": sample["audio"],
            "text": f"[espnet] {sample['text']}",
        }


class RecordingESPnetPreprocessor(AbsPreprocessor):
    def __init__(self):
        super().__init__(train=False)
        self.calls = []

    def __call__(self, uid, sample):
        self.calls.append(uid)
        return {
            "audio": sample["audio"],
            "text": f"[espnet:{uid}] {sample['text']}",
        }


class TrainFlagRecordingPreprocessor(AbsPreprocessor):
    """Records the train flag value at call time so tests can assert it."""

    def __init__(self, train: bool = False):
        super().__init__(train=train)

    def __call__(self, uid, sample):
        return {**sample, "was_train": self.train}


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


class DummyStringKeyDataset:
    def __init__(self, path=None):
        self.data = {
            "utt0": {"audio": np.random.random(16000), "text": "hello"},
            "utt1": {"audio": np.random.random(16000), "text": "world"},
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, str):
            raise KeyError("This dataset expects string-based utterance IDs.")
        return self.data[idx]

    def keys(self):
        return self.data.keys()


class DummyShardedDataset(ShardedDataset):
    def __init__(
        self,
        path=None,
        shard_id: int = 0,
        total_shards: int = 2,
        dist_world_size: int = 1,
    ):
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.dist_world_size = dist_world_size
        self.data = [
            {"audio": np.random.random(16000), "text": f"shard{shard_id}_hello"},
            {"audio": np.random.random(16000), "text": f"shard{shard_id}_world"},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def shard(self, idx):
        return DummyShardedDataset(
            shard_id=idx,
            total_shards=self.total_shards,
            dist_world_size=self.dist_world_size,
        )


class DummyBrokenShardedDataset(ShardedDataset):
    def __init__(self, total_shards=None, dist_world_size=None):
        self.total_shards = total_shards
        self.dist_world_size = dist_world_size

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError

    def shard(self, idx):
        return self


class DummyOverrideDataset(DummyDataset):
    def __init__(self, path=None):
        self.data = [
            {"audio": np.random.random(16000), "text": "override"},
            {"audio": np.random.random(16000), "text": "winner"},
        ]


def _entry(name: str, *, transform: bool = False, data_src: str = DUMMY_DATA_SRC):
    entry = {
        "name": name,
        "data_src": data_src,
        "data_src_args": {},
    }
    if transform:
        entry["transform"] = {"_target_": DUMMY_TRANSFORM_TARGET}
    return entry


# Fixtures
@pytest.fixture
def dummy_dataset_config():
    config = {
        "train": [_entry("train_dummy", transform=True)],
        "valid": [_entry("valid_dummy")],
        "test": [_entry("test_dummy")],
    }
    config = OmegaConf.create(config)
    return config


@pytest.fixture(autouse=True)
def patch_dataset_reference(monkeypatch):
    def _instantiate_dataset_reference(config, recipe_dir=None):
        plain = (
            OmegaConf.to_container(config, resolve=False)
            if OmegaConf.is_config(config)
            else dict(config)
        )
        dataset = plain.get("dataset")
        if dataset is not None:
            if isinstance(dataset, (dict, DictConfig)):
                return instantiate(dataset)
            return dataset
        if plain.get("data_src") == DUMMY_SHARDED_DATA_SRC:
            return DummyShardedDataset()
        return DummyDataset()

    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        _instantiate_dataset_reference,
    )


# Tests
def test_combined_dataset():
    ds1 = DummyDataset()
    ds2 = DummyDataset()
    combined = CombinedDataset(
        [ds1, ds2],
        [
            (DummyTransform(), do_nothing),
            (DummyTransform(), do_nothing),
        ],
    )
    assert len(combined) == 4
    assert combined[0]["text"] == "HELLO"
    assert combined[3]["text"] == "WORLD"


def test_combined_dataset_with_string_id():
    ds = DummyStringKeyDataset()
    combined = CombinedDataset(
        [ds],
        [(do_nothing, do_nothing)],
    )
    assert len(combined) == 2
    # Access via integer index (DataLoader compatibility)
    assert combined[0]["text"] == "hello"
    assert combined[1]["text"] == "world"
    # Access via utterance ID
    assert combined["utt0"]["text"] == "hello"
    assert combined["utt1"]["text"] == "world"


def test_combined_dataset_with_missing_string_id():
    ds = DummyDataset()
    combined = CombinedDataset(
        [ds],
        [(do_nothing, do_nothing)],
    )
    with pytest.raises(ValueError, match="Utterance ID 'unknown'"):
        combined["unknown"]


def test_combined_dataset_mixed_index_types():
    numeric = DummyDataset()
    stringy = DummyStringKeyDataset()
    combined = CombinedDataset(
        [numeric, stringy],
        [
            (do_nothing, do_nothing),
            (do_nothing, do_nothing),
        ],
    )

    assert len(combined) == 4
    # First two items come from numeric dataset
    assert combined[0]["text"] == "hello"
    assert combined[1]["text"] == "world"
    # Next two items from string dataset
    assert combined[2]["text"] == "hello"
    assert combined[3]["text"] == "world"
    # Direct string lookup hits the string-backed dataset
    assert combined["utt1"]["text"] == "world"


def test_combined_dataset_duplicate_string_ids_error():
    class AnotherStringDataset(DummyStringKeyDataset):
        def __init__(self):
            super().__init__()
            self.data = {
                "utt1": {"audio": np.random.random(16000), "text": "duplicate"},
            }

    ds1 = DummyStringKeyDataset()
    ds2 = AnotherStringDataset()

    with pytest.raises(ValueError, match="Duplicate utterance ID 'utt1'"):
        CombinedDataset(
            [ds1, ds2],
            [
                (do_nothing, do_nothing),
                (do_nothing, do_nothing),
            ],
        )


def test_data_organizer_init(dummy_dataset_config):
    config = dummy_dataset_config
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        test=config["test"],
        preprocessor=DummyPreprocessor(),
    )
    assert len(organizer.train) == 2
    assert organizer.train[0]["text"] == "[dummy] HELLO"
    assert len(organizer.valid) == 2
    assert organizer.valid[0]["text"] == "[dummy] hello"
    assert "test_dummy" in organizer.test


def test_data_organizer_with_string_ids():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {
                    "_target_": (
                        "test.espnet3.components.data"
                        ".test_data_organizer.DummyStringKeyDataset"
                    )
                },
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {
                    "_target_": (
                        "test.espnet3.components.data"
                        ".test_data_organizer.DummyStringKeyDataset"
                    )
                },
            }
        ],
    }

    organizer = DataOrganizer(train=config["train"], valid=config["valid"])

    assert len(organizer.train) == 2
    assert organizer.train["utt0"]["text"] == "hello"
    assert organizer.train[0]["text"] == "hello"
    assert organizer.valid["utt1"]["text"] == "world"
    assert organizer.valid[1]["text"] == "world"


def test_data_organizer_without_test():
    config = {
        "train": [_entry("train_dummy", transform=True)],
        "valid": [_entry("valid_dummy")],
        # No "test" field
    }
    config = OmegaConf.create(config)
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=DummyPreprocessor(),
    )

    assert len(organizer.train) == 2
    assert len(organizer.valid) == 2
    assert organizer.test == {}


def test_data_organizer_test_only():
    config = {
        # No "train"
        # No "valid"
        "test": [_entry("test_dummy", transform=True)],
    }

    organizer = DataOrganizer(test=config["test"], preprocessor=DummyPreprocessor())

    assert organizer.train is None
    assert organizer.valid is None
    assert "test_dummy" in organizer.test
    assert organizer.test["test_dummy"][0]["text"] == "[dummy] HELLO"


@pytest.mark.parametrize("train_count, valid_count", [(1, 1), (2, 2)])
def test_data_organizer_train_valid_multiple(train_count, valid_count):
    config = {
        "train": [_entry(f"train{i}", transform=True) for i in range(train_count)],
        "valid": [_entry(f"valid{i}") for i in range(valid_count)],
    }
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=DummyPreprocessor(),
    )
    assert len(organizer.train) == 2 * train_count
    assert len(organizer.valid) == 2 * valid_count
    assert organizer.train[0]["text"] == "[dummy] HELLO"
    assert organizer.valid[0]["text"] == "[dummy] hello"


def test_data_organizer_test_multiple_sets():
    config = {
        "test": [
            _entry("test_clean"),
            _entry("test_other", transform=True),
        ]
    }
    organizer = DataOrganizer(test=config["test"], preprocessor=DummyPreprocessor())
    assert "test_clean" in organizer.test
    assert "test_other" in organizer.test


@pytest.mark.parametrize(
    ("test_entries", "expected_name"),
    [
        (
            [
                {
                    "name": "shared_eval",
                    "dataset": {"_target_": DUMMY_DATASET_TARGET},
                },
                {
                    "name": "shared_eval",
                    "dataset": {
                        "_target_": (
                            "test.espnet3.components.data."
                            "test_data_organizer.DummyOverrideDataset"
                        )
                    },
                },
            ],
            "shared_eval",
        ),
        (
            [
                {
                    "data_src": "shared/asr",
                    "dataset": {"_target_": DUMMY_DATASET_TARGET},
                },
                {
                    "data_src": "shared/asr",
                    "dataset": {
                        "_target_": (
                            "test.espnet3.components.data."
                            "test_data_organizer.DummyOverrideDataset"
                        )
                    },
                },
            ],
            "shared/asr",
        ),
        (
            [
                {"dataset": {"_target_": DUMMY_DATASET_TARGET}},
                {
                    "dataset": {
                        "_target_": (
                            "test.espnet3.components.data."
                            "test_data_organizer.DummyOverrideDataset"
                        )
                    }
                },
            ],
            "local",
        ),
    ],
)
def test_data_organizer_test_duplicate_names_last_entry_wins(
    test_entries, expected_name
):
    organizer = DataOrganizer(test=test_entries)

    assert list(organizer.test.keys()) == [expected_name]
    assert organizer.test[expected_name][0]["text"] == "override"


def test_data_organizer_transform_only():
    config = {
        "train": [_entry("train_dummy", transform=True)],
        "valid": [_entry("valid_dummy")],
    }
    organizer = DataOrganizer(train=config["train"], valid=config["valid"])
    assert organizer.train[0]["text"] == "HELLO"
    assert organizer.valid[0]["text"] == "hello"


def test_data_organizer_no_preprocessor_instantiated_config():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": DUMMY_DATASET_TARGET},
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {"_target_": DUMMY_DATASET_TARGET},
            }
        ],
    }
    organizer = DataOrganizer(
        train=OmegaConf.create(config)["train"],
        valid=OmegaConf.create(config)["valid"],
    )
    assert organizer.train[0]["text"] == "hello"
    assert organizer.valid[0]["text"] == "hello"


def test_data_organizer_no_preprocessor_config():
    config = {
        "train": [
            {
                "name": "train_dummy",
                "dataset": {"_target_": DUMMY_DATASET_TARGET},
            }
        ],
        "valid": [
            {
                "name": "valid_dummy",
                "dataset": {"_target_": DUMMY_DATASET_TARGET},
            }
        ],
    }
    organizer = DataOrganizer(
        train=instantiate(OmegaConf.create(config)["train"]),
        valid=instantiate(OmegaConf.create(config)["valid"]),
    )
    assert organizer.train[0]["text"] == "hello"
    assert organizer.valid[0]["text"] == "hello"


def test_data_organizer_preprocessor_only():
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
    }
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=DummyPreprocessor(),
    )
    assert organizer.train[0]["text"] == "[dummy] hello"
    assert organizer.valid[0]["text"] == "[dummy] hello"


def test_data_organizer_accepts_raw_configs():
    config = OmegaConf.create(
        {
            "train": [_entry("train_dummy", transform=True)],
            "valid": [_entry("valid_dummy")],
            "preprocessor": {"_target_": DUMMY_PREPROCESSOR_TARGET},
        }
    )
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=config["preprocessor"],
    )
    assert organizer.train[0]["text"] == "[dummy] HELLO"
    assert organizer.valid[0]["text"] == "[dummy] hello"


def test_data_organizer_split_preprocessor_uses_shared_fallback(caplog):
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
        "test": [_entry("test_dummy")],
    }
    preprocessor_cfg = OmegaConf.create(
        {
            "_target_": SPLIT_RECORDING_PREPROCESSOR_TARGET,
            "tag": "shared",
            "train": {"tag": "train"},
        }
    )
    with caplog.at_level(
        logging.WARNING,
        logger="espnet3.components.data.data_organizer",
    ):
        organizer = DataOrganizer(
            train=config["train"],
            valid=config["valid"],
            test=config["test"],
            preprocessor=preprocessor_cfg,
        )
    assert organizer.train[0]["text"] == "[train] hello"
    assert organizer.valid[0]["text"] == "[shared] hello"
    assert organizer.test["test_dummy"][0]["text"] == "[shared] hello"
    assert any("missing 'test'" in r.message for r in caplog.records)


def test_data_organizer_split_preprocessor_uses_valid_for_test_by_default(caplog):
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
        "test": [_entry("test_dummy")],
    }
    preprocessor_cfg = OmegaConf.create(
        {
            "train": {
                "_target_": SPLIT_RECORDING_PREPROCESSOR_TARGET,
                "tag": "train",
            },
            "valid": {
                "_target_": SPLIT_RECORDING_PREPROCESSOR_TARGET,
                "tag": "valid",
            },
        }
    )
    with caplog.at_level(
        logging.WARNING,
        logger="espnet3.components.data.data_organizer",
    ):
        organizer = DataOrganizer(
            train=config["train"],
            valid=config["valid"],
            test=config["test"],
            preprocessor=preprocessor_cfg,
        )
    assert organizer.train[0]["text"] == "[train] hello"
    assert organizer.valid[0]["text"] == "[valid] hello"
    assert organizer.test["test_dummy"][0]["text"] == "[valid] hello"
    assert any("missing 'test'" in r.message for r in caplog.records)


def test_data_organizer_split_preprocessor_allows_test_override():
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
        "test": [_entry("test_dummy")],
    }
    preprocessor_cfg = OmegaConf.create(
        {
            "train": {
                "_target_": SPLIT_RECORDING_PREPROCESSOR_TARGET,
                "tag": "train",
            },
            "valid": {
                "_target_": SPLIT_RECORDING_PREPROCESSOR_TARGET,
                "tag": "valid",
            },
            "test": {
                "_target_": SPLIT_RECORDING_PREPROCESSOR_TARGET,
                "tag": "test",
            },
        }
    )
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        test=config["test"],
        preprocessor=preprocessor_cfg,
    )
    assert organizer.test["test_dummy"][0]["text"] == "[test] hello"


def test_data_organizer_passes_recipe_dir_to_ref_less_entries(monkeypatch):
    captured = []

    def _instantiate_dataset_reference(config, recipe_dir=None):
        captured.append((dict(config), recipe_dir))
        return DummyDataset()

    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        _instantiate_dataset_reference,
    )

    organizer = DataOrganizer(
        train=[{"data_src_args": {"split": "train"}}],
        valid=[{"data_src_args": {"split": "valid"}}],
        recipe_dir="/tmp/local_recipe",
    )

    assert len(organizer.train) == 2
    assert len(organizer.valid) == 2
    assert captured == [
        ({"data_src_args": {"split": "train"}}, "/tmp/local_recipe"),
        ({"data_src_args": {"split": "valid"}}, "/tmp/local_recipe"),
    ]


def test_data_organizer_transform_and_preprocessor():
    config = {
        "train": [_entry("train_dummy", transform=True)],
        "valid": [_entry("valid_dummy")],
    }
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=DummyPreprocessor(),
    )
    assert organizer.train[0]["text"] == "[dummy] HELLO"
    assert organizer.valid[0]["text"] == "[dummy] hello"


def test_data_organizer_train_only_assertion():
    config = {
        "train": [_entry("train_dummy")]
        # valid is missing
    }
    with pytest.raises(RuntimeError):
        DataOrganizer(train=config["train"])


def test_data_organizer_valid_only_assertion():
    config = {
        "valid": [_entry("valid_dummy")]
        # train is missing
    }
    with pytest.raises(RuntimeError):
        DataOrganizer(valid=config["valid"])


def test_data_organizer_train_and_test_without_valid_assertion():
    config = {
        "train": [_entry("train_dummy")],
        "test": [_entry("test_dummy")],
        # valid is missing
    }
    with pytest.raises(RuntimeError):
        DataOrganizer(train=config["train"], test=config["test"])


def test_data_organizer_empty_train_valid_ok():
    config = {
        "train": [],
        "valid": [],
    }
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
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
                (do_nothing, do_nothing),
                (do_nothing, do_nothing),
            ],
        )


def test_data_organizer_transform_none():
    class BrokenTransform:
        def __call__(self, sample):
            raise ValueError("Broken")

    ds = DummyDataset()
    with pytest.raises(ValueError):
        CombinedDataset([ds], [(BrokenTransform(), do_nothing)])


def test_combined_dataset_allows_missing_preprocessor():
    ds = DummyDataset()
    combined = CombinedDataset([ds], [(do_nothing, None)])
    assert combined[0]["text"] == "hello"


def test_dataset_with_transform_allows_missing_preprocessor():
    ds = DummyDataset()
    wrapped = DatasetWithTransform(ds, do_nothing, None)
    assert wrapped[0]["text"] == "hello"


def test_data_organizer_invalid_preprocessor_type():
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
    }
    with pytest.raises(AssertionError):
        DataOrganizer(
            train=config["train"],
            valid=config["valid"],
            preprocessor="not_callable",
        )


def test_espnet_preprocessor_without_transform():
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
    }

    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=ESPnetPreprocessor(),  # ESPnet-style preprocessor
    )

    sample = organizer.train[0]
    assert sample["text"] == "[espnet] hello"


def test_espnet_preprocessor_with_transform():
    config = {
        "train": [_entry("train_dummy", transform=True)],
        "valid": [_entry("valid_dummy")],
    }

    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=ESPnetPreprocessor(),  # ESPnet-style preprocessor
    )

    sample = organizer.train[0]
    assert sample["text"] == "[espnet] HELLO"


def test_data_organizer_test_uses_espnet_preprocessor_uid():
    preprocessor = RecordingESPnetPreprocessor()
    organizer = DataOrganizer(
        test=[_entry("test_dummy", transform=True)],
        preprocessor=preprocessor,
    )

    sample = organizer.test["test_dummy"][0]

    assert sample["text"] == "[espnet:0] HELLO"
    assert preprocessor.calls == ["0"]


def test_combined_dataset_sharded_consistency_error():
    # One dataset is a ShardedDataset, the other is not
    ds1 = DummyShardedDataset()
    ds2 = DummyDataset()

    # This should raise a RuntimeError due to inconsistency
    with pytest.raises(
        RuntimeError, match="If any dataset is a subclass of ShardedDataset"
    ):
        CombinedDataset(
            datasets=[ds1, ds2],
            transforms=[(DummyTransform(), DummyPreprocessor())] * 2,
            use_espnet_preprocessor=False,
        )


def test_combined_dataset_shard_returns_sharded_dataset():
    ds1 = DummyShardedDataset(shard_id=0)
    ds2 = DummyShardedDataset(shard_id=0)
    combined = CombinedDataset(
        [ds1, ds2],
        [
            (DummyTransform(), do_nothing),
            (DummyTransform(), do_nothing),
        ],
    )
    sharded = combined.shard(2)
    assert len(sharded) == 4
    assert sharded[0]["text"] == "SHARD2_HELLO"


def test_combined_dataset_sharded_missing_metadata():
    ds1 = DummyBrokenShardedDataset(total_shards=None, dist_world_size=1)
    ds2 = DummyBrokenShardedDataset(total_shards=None, dist_world_size=1)
    with pytest.raises(
        RuntimeError,
        match="ShardedDataset requires total_shards and dist_world_size",
    ):
        CombinedDataset(
            datasets=[ds1, ds2],
            transforms=[(DummyTransform(), DummyPreprocessor())] * 2,
            use_espnet_preprocessor=False,
        )


def test_combined_dataset_sharded_metadata_mismatch():
    ds1 = DummyBrokenShardedDataset(total_shards=2, dist_world_size=1)
    ds2 = DummyBrokenShardedDataset(total_shards=3, dist_world_size=1)
    with pytest.raises(
        RuntimeError, match="must share the same total_shards and dist_world_size"
    ):
        CombinedDataset(
            datasets=[ds1, ds2],
            transforms=[(DummyTransform(), DummyPreprocessor())] * 2,
            use_espnet_preprocessor=False,
        )


# -----------------------------------------------------------------------
# ESPnet preprocessor train-flag auto-setting
# -----------------------------------------------------------------------


def test_espnet_preprocessor_train_flag_auto_set_from_dictconfig():
    """Config-based AbsPreprocessor: train split gets train=True, valid gets False."""
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
    }
    preprocessor_cfg = OmegaConf.create(
        {"_target_": ESPNET_TRAIN_FLAG_PREPROCESSOR_TARGET}
    )
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=preprocessor_cfg,
    )
    assert organizer.train[0]["was_train"] is True
    assert organizer.valid[0]["was_train"] is False


def test_espnet_preprocessor_train_flag_auto_set_from_plain_dict():
    """Plain-dict config for AbsPreprocessor: same auto-setting as DictConfig."""
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
    }
    preprocessor_cfg = {"_target_": ESPNET_TRAIN_FLAG_PREPROCESSOR_TARGET}
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=preprocessor_cfg,
    )
    assert organizer.train[0]["was_train"] is True
    assert organizer.valid[0]["was_train"] is False


def test_espnet_preprocessor_train_flag_auto_set_from_instance():
    """Already-instantiated AbsPreprocessor: deepcopy for train, original for valid."""
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
    }
    pp = TrainFlagRecordingPreprocessor(train=False)
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=pp,
    )
    assert organizer.train[0]["was_train"] is True
    assert organizer.valid[0]["was_train"] is False


def test_espnet_preprocessor_test_split_uses_train_false():
    """Test-only setup: AbsPreprocessor is called with train=False."""
    preprocessor_cfg = OmegaConf.create(
        {"_target_": ESPNET_TRAIN_FLAG_PREPROCESSOR_TARGET}
    )
    organizer = DataOrganizer(
        test=[_entry("test_dummy")],
        preprocessor=preprocessor_cfg,
    )
    assert organizer.test["test_dummy"][0]["was_train"] is False


def test_espnet_preprocessor_split_config_allows_missing_valid(caplog):
    """Missing valid/test configs fall back to no-op with ESPnet call shape."""
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
        "test": [_entry("test_dummy")],
    }
    preprocessor_cfg = OmegaConf.create(
        {
            "train": {"_target_": ESPNET_TRAIN_FLAG_PREPROCESSOR_TARGET},
        }
    )
    with caplog.at_level(
        logging.WARNING,
        logger="espnet3.components.data.data_organizer",
    ):
        organizer = DataOrganizer(
            train=config["train"],
            valid=config["valid"],
            test=config["test"],
            preprocessor=preprocessor_cfg,
        )
    assert organizer.train[0]["was_train"] is True
    assert organizer.valid[0]["was_train"] is False
    assert organizer.test["test_dummy"][0]["was_train"] is False
    assert any("missing 'valid'" in r.message for r in caplog.records)
    assert any("missing 'test'" in r.message for r in caplog.records)


def test_data_organizer_split_preprocessor_missing_train_warns(caplog):
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
    }
    preprocessor_cfg = OmegaConf.create(
        {
            "valid": {
                "_target_": SPLIT_RECORDING_PREPROCESSOR_TARGET,
                "tag": "valid",
            },
        }
    )
    with caplog.at_level(
        logging.WARNING,
        logger="espnet3.components.data.data_organizer",
    ):
        organizer = DataOrganizer(
            train=config["train"],
            valid=config["valid"],
            preprocessor=preprocessor_cfg,
        )
    assert organizer.train[0]["text"] == "hello"
    assert organizer.valid[0]["text"] == "[valid] hello"
    assert any("missing 'train'" in r.message for r in caplog.records)
    assert any("missing 'test'" in r.message for r in caplog.records)


def test_custom_preprocessor_same_instance_for_all_splits():
    """Non-AbsPreprocessor: the exact same callable is used for train and valid."""
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
    }
    pp = DummyPreprocessor()
    organizer = DataOrganizer(
        train=config["train"],
        valid=config["valid"],
        preprocessor=pp,
    )
    _, (_, train_pp) = next(
        (d, t) for d, t in zip(organizer.train.datasets, organizer.train.transforms)
    )
    _, (_, valid_pp) = next(
        (d, t) for d, t in zip(organizer.valid.datasets, organizer.valid.transforms)
    )
    assert train_pp is pp
    assert valid_pp is pp


def test_espnet_preprocessor_explicit_train_in_config_warns(caplog):
    """Explicit 'train' key in ESPnet preprocessor config emits a warning."""
    config = {
        "train": [_entry("train_dummy")],
        "valid": [_entry("valid_dummy")],
    }
    preprocessor_cfg = OmegaConf.create(
        {
            "_target_": ESPNET_TRAIN_FLAG_PREPROCESSOR_TARGET,
            "train": False,
        }
    )
    with caplog.at_level(
        logging.WARNING,
        logger="espnet3.components.data.data_organizer",
    ):
        organizer = DataOrganizer(
            train=config["train"],
            valid=config["valid"],
            preprocessor=preprocessor_cfg,
        )
    assert any("train" in r.message.lower() for r in caplog.records)
    # Flag is still auto-set correctly despite the explicit config value.
    assert organizer.train[0]["was_train"] is True
    assert organizer.valid[0]["was_train"] is False
