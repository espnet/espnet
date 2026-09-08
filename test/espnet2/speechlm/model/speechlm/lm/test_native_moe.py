"""Exercise native MoE dispatch and router gradients with small local models."""

import copy
from unittest.mock import patch

import pytest
import torch

qwen3_moe = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")

from espnet2.speechlm.model.speechlm.lm.parallel import (  # noqa: E402
    configure_moe_model,
)
from espnet2.speechlm.model.speechlm.lm.parallel_pp import (  # noqa: E402
    build_parallel_pp_hf_class,
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
    if not hasattr(torch.nn.functional, "grouped_mm"):
        pytest.skip("Native grouped MM requires PyTorch 2.10 or newer")

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
        torch.nn.functional,
        "grouped_mm",
        wraps=torch.nn.functional.grouped_mm,
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


@pytest.mark.parametrize("ac_mode", [None, "full", "moe"])
def test_pp_replica_preserves_router_loss_and_gradients(moe_model, tmp_path, ac_mode):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    moe_model.config.save_pretrained(tmp_path)
    pp_cls = build_parallel_pp_hf_class(str(tmp_path))
    vocab_size = moe_model.config.vocab_size
    vocab_meta = {
        "vocab_size": vocab_size,
        "num_stream": 1,
        "mm_start": vocab_size,
        "mm_end": vocab_size,
        "vocab_weight": torch.ones(vocab_size),
        "vocab": [],
        "vocab_intervals": {},
    }
    replica = pp_cls._empty_init(str(tmp_path), {}, vocab_meta)
    assert all(parameter.is_meta for parameter in replica.parameters())
    gates = []
    for layer in replica.model.layers:
        assert layer.mlp.experts.config._experts_implementation == "grouped_mm"
        gates.append(layer.mlp.gate)
    # CPU reference checks routing and autograd; the CUDA test checks dispatch.
    replica.config._experts_implementation = "eager"

    replica.model.load_state_dict(moe_model.model.state_dict(), assign=True)
    replica.model.rotary_emb = copy.deepcopy(moe_model.model.rotary_emb)
    if ac_mode == "full":
        for i, layer in enumerate(replica.model.layers):
            replica.model.layers[i] = checkpoint_wrapper(layer)
    elif ac_mode == "moe":
        for layer in replica.model.layers:
            layer.mlp = checkpoint_wrapper(layer.mlp)

    inputs = torch.randn(2, 8, moe_model.config.hidden_size)
    position_ids = torch.arange(8).expand(2, -1)
    reference = moe_model.model(
        inputs_embeds=inputs,
        position_ids=position_ids,
        output_router_logits=True,
        use_cache=False,
    )
    expected_logits = torch.cat(reference.router_logits, dim=0)
    expected_loss = replica.load_balancing_loss_func(
        reference.router_logits, num_experts=4, top_k=2
    )
    expected_gradients = torch.autograd.grad(
        expected_loss, [layer.mlp.gate.weight for layer in moe_model.model.layers]
    )

    for _ in range(2):
        _, logits = replica._run_decoder_layers(inputs, position_ids)
        torch.testing.assert_close(logits, expected_logits)
        loss = replica.load_balancing_loss_func((logits,), num_experts=4, top_k=2)
        torch.testing.assert_close(loss, expected_loss)
        gradients = torch.autograd.grad(loss, [gate.weight for gate in gates])
        for actual, expected in zip(gradients, expected_gradients):
            assert torch.isfinite(actual).all()
            assert actual.abs().sum() > 0
            torch.testing.assert_close(actual, expected)
        assert all(len(gate._forward_hooks) == 1 for gate in gates)
