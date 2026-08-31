"""Tests for the PEFT helper utilities."""

import pytest
import torch
import torch.nn as nn

from espnet2.layers.lora import Linear as LoraLinear
from espnet2.layers.lora import lora_state_dict, mark_only_lora_as_trainable


def _build_model():
    return nn.Sequential(
        LoraLinear(8, 8, r=2, lora_alpha=2, bias=True),
        nn.ReLU(),
        nn.Linear(8, 4, bias=True),
    )


def test_mark_only_lora_freezes_non_lora_params():
    model = _build_model()
    mark_only_lora_as_trainable(model, bias="none")
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert p.requires_grad, f"{n} should remain trainable"
        else:
            assert not p.requires_grad, f"{n} should be frozen"


def test_mark_only_lora_bias_all():
    model = _build_model()
    mark_only_lora_as_trainable(model, bias="all")
    for n, p in model.named_parameters():
        if "lora_" in n or "bias" in n:
            assert p.requires_grad, f"{n} should be trainable under bias='all'"
        else:
            assert not p.requires_grad, f"{n} should be frozen under bias='all'"


def test_mark_only_lora_bias_lora_only_keeps_only_adapted_biases():
    model = _build_model()
    mark_only_lora_as_trainable(model, bias="lora_only")
    # The LoRA-adapted module's bias is trainable; the plain Linear's is not.
    assert model[0].bias.requires_grad is True
    assert model[2].bias.requires_grad is False


def test_mark_only_lora_unknown_bias_raises():
    model = _build_model()
    with pytest.raises(NotImplementedError):
        mark_only_lora_as_trainable(model, bias="unsupported")


def test_lora_state_dict_returns_only_adapter_keys():
    model = _build_model()
    sd = lora_state_dict(model, bias="none")
    assert sd, "state dict should not be empty"
    assert all("lora_" in k for k in sd)


def test_lora_state_dict_bias_all_includes_bias_keys():
    model = _build_model()
    sd = lora_state_dict(model, bias="all")
    assert any("lora_" in k for k in sd)
    assert any(k.endswith("bias") for k in sd)


def test_lora_state_dict_bias_lora_only_includes_adapted_bias_only():
    model = _build_model()
    sd = lora_state_dict(model, bias="lora_only")
    bias_keys = [k for k in sd if k.endswith("bias")]
    # Only the LoRA-adapted module's bias should be present.
    assert bias_keys == ["0.bias"], f"unexpected bias keys: {bias_keys}"


def test_lora_state_dict_unknown_bias_raises():
    model = _build_model()
    with pytest.raises(NotImplementedError):
        lora_state_dict(model, bias="unsupported")


def test_state_dict_roundtrip_lora_only():
    """Re-applying a saved LoRA state dict should not change the model output."""
    torch.manual_seed(0)
    model = _build_model()
    mark_only_lora_as_trainable(model, bias="none")
    model.eval()
    x = torch.randn(2, 8)
    before = model(x)

    sd = lora_state_dict(model, bias="none")
    # Zero out lora params then reload.
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.zero_()
    model.load_state_dict(sd, strict=False)
    after = model(x)
    assert torch.allclose(before, after, atol=1e-6)
