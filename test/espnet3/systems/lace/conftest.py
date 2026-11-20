"""Shared fixtures for the espnet3 LACE system tests."""

from test.espnet3.systems.lace._gan_dummies import DummyDataset

import pytest

from espnet3.components.data import data_organizer as data_organizer_module


@pytest.fixture(autouse=True)
def patch_dataset_reference(monkeypatch):
    """Serve ``DummyDataset`` for every dataset reference the dummies' config names."""
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: DummyDataset(),
    )
