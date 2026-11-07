"""Tests for the PEFT helper utilities."""

import pytest
import torch
import torch.nn as nn

from espnet2.layers.lora import LINEAR_BACKENDS
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
    """Re-applying a saved LoRA state dict should restore the model output."""
    torch.manual_seed(0)
    model = _build_model()
    mark_only_lora_as_trainable(model, bias="none")
    # Stay unmerged (train mode) and give the adapters a nonzero contribution
    # so a failed restore is observable in the output.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_()
    x = torch.randn(2, 8)
    before = model(x)

    # Clone the saved tensors: state_dict() holds references, so zeroing the
    # parameters below would otherwise also zero the "saved" values.
    sd = {k: v.detach().clone() for k, v in lora_state_dict(model, bias="none").items()}
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.zero_()
    zeroed = model(x)
    assert not torch.allclose(before, zeroed, atol=1e-6), "adapters had no effect"
    model.load_state_dict(sd, strict=False)
    after = model(x)
    assert torch.allclose(before, after, atol=1e-6)


@pytest.mark.parametrize("name,cls", sorted(LINEAR_BACKENDS.items()))
def test_lora_state_dict_covers_every_backend(name, cls):
    """`save_strategy: adapter_only` must not silently save an empty dict.

    SVFT trains `m_entries`/`gate` and SSVD trains `s`/`gate`/`K_vec`; none of
    those contain the substring "lora_", so a name-based filter drops them.
    """
    layer = cls(8, 8, r=2, lora_alpha=2, bias=True)
    model = nn.Sequential(layer, nn.Linear(8, 4, bias=True))
    sd = lora_state_dict(model, bias="none")
    assert sd, f"{name}: lora_state_dict() returned nothing to save"
    trainable = {
        n for n, p in model.named_parameters() if p.requires_grad and "0." in n
    }
    trainable -= {"0.weight", "0.bias"}
    missing = trainable - set(sd)
    assert not missing, f"{name}: adapter parameters missing from the dict: {missing}"
    assert "1.weight" not in sd, f"{name}: a non-adapter parameter leaked in"


@pytest.mark.parametrize("name,cls", sorted(LINEAR_BACKENDS.items()))
def test_mark_only_lora_as_trainable_covers_every_backend(name, cls):
    layer = cls(8, 8, r=2, lora_alpha=2, bias=True)
    model = nn.Sequential(layer, nn.Linear(8, 4, bias=True))
    mark_only_lora_as_trainable(model, bias="none")
    assert any(
        p.requires_grad for p in layer.parameters()
    ), f"{name}: every adapter parameter was frozen"
    assert not model[1].weight.requires_grad
