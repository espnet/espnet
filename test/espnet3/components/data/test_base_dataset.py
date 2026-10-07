"""Tests for espnet3.components.data.base_dataset.BaseDataset."""

import numpy as np
import pytest

from espnet3.api.inference import Field
from espnet3.components.contract.dataset import DatasetContractError
from espnet3.components.data.base_dataset import BaseDataset


def test_base_dataset_checks_only_the_first_getitem_call():
    calls = []

    class Recipe(BaseDataset):
        fields = (Field("speech", "audio"), Field("text", "text"))

        def __getitem__(self, index):
            calls.append(index)
            return {"speech": np.zeros(16000, dtype=np.float32), "text": "hi"}

    recipe = Recipe()
    assert recipe[0]["text"] == "hi"
    assert recipe[1]["text"] == "hi"
    assert calls == [0, 1]


def test_base_dataset_rejects_item_missing_a_declared_field():
    class Recipe(BaseDataset):
        fields = (Field("speech", "audio"), Field("text", "text"))

        def __getitem__(self, index):
            return {"text": "hi"}

    with pytest.raises(DatasetContractError, match="lacks declared field 'speech'"):
        Recipe()[0]


def test_base_dataset_accepts_instance_level_fields():
    class Recipe(BaseDataset):
        def __init__(self, extra):
            self.extra = extra
            self.fields = (Field("speech", "audio"), Field(extra, "text"))

        def __getitem__(self, index):
            return {"speech": np.zeros(16000, dtype=np.float32), self.extra: "hi"}

    recipe = Recipe("speaker")
    assert recipe[0]["speaker"] == "hi"


def test_base_dataset_requires_fields_somewhere():
    class Recipe(BaseDataset):
        def __getitem__(self, index):
            return {"text": "hi"}

    with pytest.raises(TypeError, match="does not declare fields"):
        Recipe()[0]


def test_base_dataset_rejects_malformed_class_fields_at_class_creation():
    with pytest.raises(TypeError, match="must name at least one field"):

        class Recipe(BaseDataset):
            fields = ()

            def __getitem__(self, index):
                return {}


def test_base_dataset_requires_getitem_override():
    with pytest.raises(TypeError, match="abstract"):
        BaseDataset()


def test_base_dataset_preserves_getitem_name_and_docstring():
    class Recipe(BaseDataset):
        fields = (Field("text", "text"),)

        def __getitem__(self, index):
            """Return one item."""
            return {"text": "hi"}

    assert Recipe.__getitem__.__name__ == "__getitem__"
    assert Recipe.__getitem__.__doc__ == "Return one item."


def test_base_dataset_checks_only_once_through_a_grandchild(monkeypatch):
    import espnet3.components.data.base_dataset as base_dataset_module

    calls = []
    real_check_item = base_dataset_module.check_item

    def spy(*args, **kwargs):
        calls.append(1)
        return real_check_item(*args, **kwargs)

    monkeypatch.setattr(base_dataset_module, "check_item", spy)

    class Recipe(BaseDataset):
        fields = (Field("text", "text"),)

        def __getitem__(self, index):
            return {"text": "hi"}

    class Grandchild(Recipe):
        pass

    Grandchild()[0]
    assert len(calls) == 1
