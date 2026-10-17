"""Opt-in GPU integration check for SpeechLM native MoE dispatch.

Run from the repository root in a SpeechLM environment on a CUDA SM80+ GPU:
    python -m pytest -o addopts= test_utils/speechlm/test_native_moe_cuda.py

This file is outside the default CPU-only ``test/`` collection. It creates
a small model locally and compares outputs and gradients with eager experts.
"""

import copy
from unittest.mock import patch

import pytest
import torch

qwen3_moe = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")

from espnet2.speechlm.model.speechlm.lm.parallel import (  # noqa: E402
    configure_moe_model,
)


@pytest.fixture
def moe_model():
    torch.manual_seed(0)
    config = qwen3_moe.Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        attn_implementation="sdpa",
        experts_implementation="eager",
    )
    return qwen3_moe.Qwen3MoeForCausalLM(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_grouped_mm_matches_eager(moe_model):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Native grouped MM requires SM80 or newer")
    if hasattr(torch.nn.functional, "grouped_mm"):
        grouped_mm_owner, grouped_mm_name = torch.nn.functional, "grouped_mm"
    elif hasattr(torch, "_grouped_mm"):
        grouped_mm_owner, grouped_mm_name = torch, "_grouped_mm"
    else:
        pytest.skip("No native grouped-MM callable is available")

    model = configure_moe_model(moe_model).to(device="cuda", dtype=torch.bfloat16)
    grouped = model.model.layers[0].mlp
    eager = copy.deepcopy(grouped)
    eager.experts.config._experts_implementation = "eager"
    x = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    x_grouped = x.clone().requires_grad_()
    x_eager = x.clone().requires_grad_()
    expected = eager(x_eager)
    expected.float().square().sum().backward()

    with patch.object(
        grouped_mm_owner,
        grouped_mm_name,
        wraps=getattr(grouped_mm_owner, grouped_mm_name),
    ) as grouped_mm:
        actual = grouped(x_grouped)
        actual.float().square().sum().backward()

    assert grouped_mm.call_count == 2
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-2)
    torch.testing.assert_close(x_grouped.grad, x_eager.grad, atol=2e-5, rtol=5e-2)
    for parameter, reference in zip(grouped.parameters(), eager.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-5, rtol=5e-2)
