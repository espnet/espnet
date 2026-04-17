"""Tests for espnet2/speechlm/model/speechlm/parallel_utils/grouped_moe.py.

The grouped MoE module depends on torchtitan (DeepEP) and HF Qwen3 MoE.
All tests use pytest.importorskip so they skip in minimal environments and
run when those dependencies are installed. ``torch._grouped_mm`` is a CUDA
+ bfloat16 kernel, so numeric parity tests also require CUDA.
"""

import pytest
import torch

pytest.importorskip(
    "torchtitan.distributed.deepep",
    reason="torchtitan DeepEP not installed",
)
qwen3_moe_mod = pytest.importorskip(
    "transformers.models.qwen3_moe.modeling_qwen3_moe",
    reason="transformers Qwen3 MoE not available",
)

from espnet2.speechlm.model.speechlm.parallel_utils.grouped_moe import (  # noqa: E402
    GroupedMoeBlock,
    _route,
)

Qwen3MoeConfig = qwen3_moe_mod.Qwen3MoeConfig
Qwen3MoeSparseMoeBlock = qwen3_moe_mod.Qwen3MoeSparseMoeBlock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_moe_config(
    hidden_size=16,
    intermediate_size=32,
    num_experts=4,
    top_k=2,
    norm_topk_prob=True,
):
    """Build a small Qwen3MoeConfig suitable for MoE-block construction."""
    return Qwen3MoeConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=intermediate_size,
        norm_topk_prob=norm_topk_prob,
        hidden_act="silu",
    )


def _reference_forward(block: Qwen3MoeSparseMoeBlock, x: torch.Tensor):
    """Reference MoE forward using a Python loop over individual experts.

    Mirrors the math of ``GroupedMoeBlock`` without the grouped_mm fusion.
    Returns ``(output, router_logits)``.
    """
    bsz, seq_len, hidden_dim = x.shape
    flat = x.view(-1, hidden_dim)
    router_logits, routing_weights, selected_experts = _route(
        block.gate, flat, block.top_k, block.norm_topk_prob
    )
    out = torch.zeros_like(flat)
    for expert_idx, expert in enumerate(block.experts):
        mask = selected_experts == expert_idx  # (T, K)
        if not mask.any():
            continue
        token_positions, topk_positions = mask.nonzero(as_tuple=True)
        expert_in = flat[token_positions]
        expert_out = expert(expert_in)
        weights = routing_weights[token_positions, topk_positions].unsqueeze(-1)
        out.index_add_(0, token_positions, expert_out * weights)
    return out.view(bsz, seq_len, hidden_dim), router_logits


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_route_topk_shape_and_normalization():
    torch.manual_seed(0)
    num_tokens, hidden, num_experts, top_k = 8, 12, 4, 2
    gate = torch.nn.Linear(hidden, num_experts, bias=False)
    hidden_states = torch.randn(num_tokens, hidden)

    logits, weights, experts = _route(
        gate, hidden_states, top_k=top_k, norm_topk_prob=True
    )

    assert logits.shape == (num_tokens, num_experts)
    assert weights.shape == (num_tokens, top_k)
    assert experts.shape == (num_tokens, top_k)
    assert experts.dtype == torch.int64
    assert (experts >= 0).all() and (experts < num_experts).all()

    torch.testing.assert_close(
        weights.float().sum(dim=-1),
        torch.ones(num_tokens),
        rtol=1e-5,
        atol=1e-5,
    )


def test_route_topk_no_normalization_keeps_raw_weights():
    torch.manual_seed(0)
    gate = torch.nn.Linear(8, 6, bias=False)
    hidden_states = torch.randn(5, 8)
    _, weights, _ = _route(gate, hidden_states, top_k=2, norm_topk_prob=False)

    sums = weights.float().sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# GroupedMoeBlock parity (EP=1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="torch._grouped_mm requires CUDA",
)
def test_grouped_moe_block_forward_parity():
    """GroupedMoeBlock output matches reference per-expert loop (EP=1)."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config = _make_moe_config(
        hidden_size=16, intermediate_size=32, num_experts=4, top_k=2
    )
    original = Qwen3MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
    original.eval()

    grouped = GroupedMoeBlock(original).to(device=device, dtype=dtype)
    grouped.eval()

    x = torch.randn(2, 3, config.hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        ref_out, ref_logits = _reference_forward(original, x)
        grouped_out, grouped_logits = grouped(x)

    torch.testing.assert_close(grouped_logits, ref_logits, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(grouped_out, ref_out, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="torch._grouped_mm requires CUDA",
)
def test_grouped_moe_block_preserves_frozen_state():
    """GroupedExperts preserves requires_grad from source experts."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config = _make_moe_config(num_experts=4, top_k=2)
    original = Qwen3MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
    for p in original.experts.parameters():
        p.requires_grad = False

    grouped = GroupedMoeBlock(original).to(device=device, dtype=dtype)
    for name, p in grouped.experts.named_parameters():
        assert not p.requires_grad, f"{name} should be frozen"
