"""Tests for the callback that seeds a run from a trained checkpoint."""

from types import SimpleNamespace

import pytest
import torch

from espnet3.components.callbacks.pretrained_parameters import (
    PretrainedParametersCallback,
)


class _Tiny(torch.nn.Module):
    """A two-part model, so a partial load can be told from a whole one."""

    def __init__(self, fill: float = 0.0):
        """Fill both parts with ``fill`` so a copied value is recognisable."""
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2, bias=False)
        self.decoder = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.encoder.weight.fill_(fill)
            self.decoder.weight.fill_(fill)


@pytest.fixture()
def checkpoint(tmp_path):
    """Write a checkpoint whose parameters are all ones."""
    path = tmp_path / "trained.pth"
    torch.save(_Tiny(fill=1.0).state_dict(), path)
    return path


def _module(model):
    """A stand-in for the LightningModule the callback reads ``model`` off."""
    return SimpleNamespace(model=model)


def test_init_param_accepts_a_single_string():
    """One specification need not be wrapped in a list."""
    assert PretrainedParametersCallback(init_param="a.pth").init_param == ["a.pth"]


def test_init_param_accepts_a_sequence():
    """Several specifications are kept in order."""
    callback = PretrainedParametersCallback(init_param=["a.pth", "b.pth"])
    assert callback.init_param == ["a.pth", "b.pth"]


def test_setup_loads_the_whole_checkpoint(checkpoint):
    """A bare path copies every parameter."""
    model = _Tiny(fill=0.0)
    PretrainedParametersCallback(init_param=str(checkpoint)).setup(
        trainer=None, pl_module=_module(model), stage="fit"
    )
    assert torch.equal(model.encoder.weight, torch.ones(2, 2))
    assert torch.equal(model.decoder.weight, torch.ones(2, 2))


def test_setup_loads_only_the_named_part(checkpoint):
    """`<path>:<src>:<dst>` is the egs2 `--pretrained_model` syntax.

    This is how the recipe starts a two-pass model from the first pass's
    encoder, so the decoder must be left as initialised.
    """
    model = _Tiny(fill=0.0)
    PretrainedParametersCallback(init_param=f"{checkpoint}:encoder:encoder").setup(
        trainer=None, pl_module=_module(model), stage="fit"
    )
    assert torch.equal(model.encoder.weight, torch.ones(2, 2))
    assert torch.equal(model.decoder.weight, torch.zeros(2, 2))


@pytest.mark.parametrize("stage", ["validate", "test", "predict"])
def test_setup_loads_nothing_outside_fit(checkpoint, stage):
    """Only a fit run starts from pretrained parameters."""
    model = _Tiny(fill=0.0)
    PretrainedParametersCallback(init_param=str(checkpoint)).setup(
        trainer=None, pl_module=_module(model), stage=stage
    )
    assert torch.equal(model.encoder.weight, torch.zeros(2, 2))


def test_setup_loads_once(checkpoint):
    """A second setup must not overwrite what training has since learned."""
    model = _Tiny(fill=0.0)
    callback = PretrainedParametersCallback(init_param=str(checkpoint))
    callback.setup(trainer=None, pl_module=_module(model), stage="fit")
    with torch.no_grad():
        model.encoder.weight.fill_(2.0)
    callback.setup(trainer=None, pl_module=_module(model), stage="fit")
    assert torch.equal(model.encoder.weight, torch.full((2, 2), 2.0))


def test_setup_leaves_requires_grad_alone(checkpoint):
    """Loading composes with `freeze_param`; it copies values only."""
    model = _Tiny(fill=0.0)
    model.encoder.weight.requires_grad_(False)
    PretrainedParametersCallback(init_param=str(checkpoint)).setup(
        trainer=None, pl_module=_module(model), stage="fit"
    )
    assert model.encoder.weight.requires_grad is False
    assert torch.equal(model.encoder.weight, torch.ones(2, 2))
