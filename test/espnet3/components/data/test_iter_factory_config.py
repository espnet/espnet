"""Hydra container conversion at the ESPnet3 iterator construction boundary."""

import pytest
from omegaconf import OmegaConf

from espnet3.components.data import dataloader as module


class ListFactory:
    """Require the same built-in container types as native ESPnet2 factories."""

    def __init__(self, dataset, batches, excluded_key_prefixes):
        if not isinstance(excluded_key_prefixes, list) or not all(
            isinstance(s, str) for s in excluded_key_prefixes
        ):
            raise TypeError("excluded_key_prefixes must be list[str]")
        self.dataset = dataset

    def build_iter(self, epoch):
        return iter(self.dataset)


@pytest.mark.parametrize("prefixes,valid", [(["enroll_ref"], True), ([12], False)])
def test_factory_receives_builtin_containers(monkeypatch, prefixes, valid):
    monkeypatch.setattr(module, "build_batch_sampler", lambda **kwargs: [["a"]])
    config = OmegaConf.create(
        {
            "_target_": f"{__name__}.ListFactory",
            "excluded_key_prefixes": prefixes,
            "batches": {},
        }
    )
    builder = module.DataLoaderBuilder(["a"], None, None, 1, 0)
    if valid:
        assert list(builder._build_iter_factory(config)) == ["a"]
    else:
        with pytest.raises(Exception, match=r"list\[str\]"):
            builder._build_iter_factory(config)
