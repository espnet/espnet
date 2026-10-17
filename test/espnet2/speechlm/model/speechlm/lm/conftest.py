"""Local model fixtures shared by SpeechLM language-model tests."""

import pytest
import torch


@pytest.fixture
def moe_model():
    """Small local Qwen3-MoE; no downloads or GPU initialization."""
    qwen3_moe = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
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
