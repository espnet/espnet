"""Fixtures for the codec system tests."""

from test.espnet3.systems.esp2_codec._gan_dummies import DummyDataset

import pytest

from espnet3.components.data import data_organizer as data_organizer_module


@pytest.fixture(autouse=True)
def patch_dataset_reference(monkeypatch):
    """Serve the dummy dataset for every dataset reference the organizer resolves."""
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: DummyDataset(),
    )
