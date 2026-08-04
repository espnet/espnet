import pytest

from espnet3.components.data import data_organizer as data_organizer_module

from ._gan_dummies import DummyDataset


@pytest.fixture(autouse=True)
def patch_dataset_reference(monkeypatch):
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: DummyDataset(),
    )
