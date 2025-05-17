from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from espnet3.trainer import HybridOptim


def create_simple_model_and_optim():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 5)
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.1)
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
    return [opt1, opt2]


def test_zero_grad_called():
    """
    H001: Check that zero_grad is called on all optimizers
    """
    opt1 = MagicMock()
    opt2 = MagicMock()
    hybrid = HybridOptim([opt1, opt2])
    hybrid.zero_grad(set_to_none=True)
    opt1.zero_grad.assert_called_once_with(set_to_none=True)
    opt2.zero_grad.assert_called_once_with(set_to_none=True)


def test_step_called_and_loss_returned():
    """
    H002: Check that step is called and closure returns loss
    """
    opt1 = MagicMock()
    opt2 = MagicMock()
    hybrid = HybridOptim([opt1, opt2])

    def closure():
        return torch.tensor(1.23)

    loss = hybrid.step(closure)
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, torch.tensor(1.23))
    opt1.step.assert_called_once()
    opt2.step.assert_called_once()


def test_state_dict_and_load_state_dict_roundtrip():
    """
    H003: Save state_dict and restore it correctly
    """
    optimizers = create_simple_model_and_optim()
    hybrid = HybridOptim(optimizers)

    # dummy gradient + step
    for p in optimizers[0].param_groups[0]["params"]:
        p.grad = torch.ones_like(p)
    hybrid.step()

    saved = hybrid.state_dict()
    hybrid.load_state_dict(saved)


def test_repr_contains_optimizer_info():
    """
    H004: __repr__ contains optimizer types
    """
    optimizers = create_simple_model_and_optim()
    hybrid = HybridOptim(optimizers)
    rep = repr(hybrid)
    assert "SGD" in rep
    assert "Adam" in rep


def test_combined_properties():
    """
    H005: param_groups, state, defaults are combined from all optimizers
    """
    optimizers = create_simple_model_and_optim()
    hybrid = HybridOptim(optimizers)

    # Make sure all properties return flattened lists/dicts
    assert isinstance(hybrid.param_groups, list)
    assert all(isinstance(pg, dict) for pg in hybrid.param_groups)

    assert isinstance(hybrid.state, dict)
    assert isinstance(hybrid.defaults, dict)
