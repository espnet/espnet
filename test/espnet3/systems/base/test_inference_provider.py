"""Model construction isolates temporary process-wide directory changes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.systems.base.inference_provider import InferenceProvider


@pytest.mark.parametrize("second_has_recipe", [False, True])
def test_concurrent_model_builds_keep_their_working_directories(
    tmp_path, monkeypatch, second_has_recipe
):
    """A second model waits until the first model restores the working directory."""
    original = Path.cwd()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_entered = Event()
    release_first = Event()
    second_started = Event()
    second_entered = Event()

    def instantiate(config, **kwargs):
        if config.name == "first":
            first_entered.set()
            assert release_first.wait(5)
        else:
            second_entered.set()
        return SimpleNamespace(directory=Path.cwd())

    monkeypatch.setattr(
        "espnet3.systems.base.inference_provider.instantiate", instantiate
    )
    first = OmegaConf.create(
        {"model": {"name": "first"}, "device": "cpu", "recipe_dir": str(first_dir)}
    )
    second = OmegaConf.create({"model": {"name": "second"}, "device": "cpu"})
    if second_has_recipe:
        second.recipe_dir = str(second_dir)

    def build_second():
        second_started.set()
        return InferenceProvider.build_model(second)

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(InferenceProvider.build_model, first)
        try:
            assert first_entered.wait(5)
            two = pool.submit(build_second)
            assert second_started.wait(5)
            assert not second_entered.wait(0.1)
        finally:
            release_first.set()
        assert one.result(timeout=5).directory == first_dir
        assert two.result(timeout=5).directory == (
            second_dir if second_has_recipe else original
        )
    assert Path.cwd() == original
