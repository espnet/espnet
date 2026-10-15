"""Dataset contracts survive removal of the inference declarations."""

import numpy as np
import pytest

from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract.data import DatasetContractError


class Samples(BaseDataset):
    fields = {"speech": np.ndarray, "text": str, "utt_id": (str, type(None))}

    def __init__(self, item):
        self.item = item

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.item


def test_dataset_checks_required_and_optional_values():
    item = {"speech": np.zeros(4), "text": "hello", "extra": 1}
    dataset = Samples(item)
    assert dataset[0] is item
    # A successful first check is cached, as for the original dataset contract.
    dataset.item = {}
    assert dataset[0] == {}


@pytest.mark.parametrize(
    "item",
    [
        None,
        {},
        {"speech": [1], "text": "a"},
        {"speech": np.zeros(4), "text": 1},
        {"speech": np.zeros(4), "text": "a", "utt_id": 1},
    ],
)
def test_dataset_rejects_invalid_item(item):
    with pytest.raises(DatasetContractError):
        Samples(item)[0]


def test_invalid_first_item_does_not_disable_validation():
    dataset = Samples({})
    for _ in range(2):
        with pytest.raises(DatasetContractError, match="speech"):
            dataset[0]


@pytest.mark.parametrize("fields", [(), {}, {"text": "str"}, {"": str}])
def test_dataset_rejects_invalid_declaration(fields):
    with pytest.raises(DatasetContractError):
        type("InvalidDataset", (Samples,), {"fields": fields})


def test_dataset_instance_can_declare_fields():
    class Dynamic(BaseDataset):
        def __init__(self):
            self.fields = {"text": str}

        def __getitem__(self, index):
            return {"text": "hello"}

    assert Dynamic()[0] == {"text": "hello"}
