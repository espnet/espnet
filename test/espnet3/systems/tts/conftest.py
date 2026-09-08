from test.espnet3.systems.tts._gan_dummies import DummyDataset

import pytest

from espnet3.components.data import data_organizer as data_organizer_module


@pytest.fixture
def patch_dataset_reference(monkeypatch):
    """Keep DataOrganizer off the filesystem when a module builds datasets."""
    monkeypatch.setattr(
        data_organizer_module,
        "instantiate_dataset_reference",
        lambda config, recipe_dir=None: DummyDataset(),
    )
