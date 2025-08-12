from unittest.mock import MagicMock

# import pytest
import torch
from torch import nn

from espnet3.trainer.multiple_optim import MultipleOptim

# ===============================================================
# Test Case Summary for MultipleOptim
# ===============================================================
#
# Basic Functionality Tests
# | Test Name                          | Description                           |
# |-----------------------------------|----------------------------------------|
# | test_zero_grad_called             | Ensures zero_grad() is called with     |
# |                                   | correct arguments on all optimizers    |
# | test_step_called_and_loss_returned| Confirms step() calls each optimizer's |
# |                                   | step and returns closure loss          |
# | test_state_dict_and_load_state_dict_roundtrip | Tests saving and restoring of      |
# |                                   | optimizer state dicts for round-trip integrity |
# | test_repr_contains_optimizer_info | Checks that __repr__ includes optimizer names  |
# |                                   | (SGD, Adam, etc.)                              |
#
# Combined Properties Tests
# | Test Name                          | Description                           |
# |-----------------------------------|----------------------------------------|
# | test_combined_properties          | Validates param_groups, state, defaults|
# |                                   | are aggregated correctly               |


def create_simple_model_and_optim():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 5)
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.1)
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
    return [opt1, opt2]


def test_zero_grad_called():
    opt1 = MagicMock()
    opt2 = MagicMock()
    mopts = MultipleOptim([opt1, opt2])
    mopts.zero_grad(set_to_none=True)
    opt1.zero_grad.assert_called_once_with(set_to_none=True)
    opt2.zero_grad.assert_called_once_with(set_to_none=True)


def test_step_called_and_loss_returned():
    opt1 = MagicMock()
    opt2 = MagicMock()
    mopts = MultipleOptim([opt1, opt2])

    def closure():
        return torch.tensor(1.23)

    loss = mopts.step(closure)
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, torch.tensor(1.23))
    opt1.step.assert_called_once()
    opt2.step.assert_called_once()


def test_state_dict_and_load_state_dict_roundtrip():
    optimizers = create_simple_model_and_optim()
    mopts = MultipleOptim(optimizers)

    # dummy gradient + step
    for p in optimizers[0].param_groups[0]["params"]:
        p.grad = torch.ones_like(p)
    mopts.step()

    saved = mopts.state_dict()
    mopts.load_state_dict(saved)


def test_repr_contains_optimizer_info():
    optimizers = create_simple_model_and_optim()
    mopts = MultipleOptim(optimizers)
    rep = repr(mopts)
    assert "SGD" in rep
    assert "Adam" in rep


def test_combined_properties():
    optimizers = create_simple_model_and_optim()
    mopts = MultipleOptim(optimizers)

    # Make sure all properties return flattened lists/dicts
    assert isinstance(mopts.param_groups, list)
    assert all(isinstance(pg, dict) for pg in mopts.param_groups)

    assert isinstance(mopts.state, dict)
    assert isinstance(mopts.defaults, dict)
