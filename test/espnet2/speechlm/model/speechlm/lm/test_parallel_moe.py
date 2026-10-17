"""CPU tests for native MoE configuration in speechlm/lm/parallel.py."""

import torch

from espnet2.speechlm.model.speechlm.lm.loss import (
    memory_efficient_load_balancing_loss,
)
from espnet2.speechlm.model.speechlm.lm.parallel import configure_moe_model


def test_configure_moe_model_preserves_native_experts(moe_model):
    experts = [layer.mlp.experts for layer in moe_model.model.layers]
    parameters = tuple(moe_model.parameters())
    assert moe_model.config._experts_implementation == "eager"

    configured = configure_moe_model(moe_model)

    assert configured is moe_model
    assert configured.load_balancing_loss_func is memory_efficient_load_balancing_loss
    for layer, original in zip(configured.model.layers, experts):
        assert layer.mlp.experts is original
        assert original.config is configured.config
        assert original.config._experts_implementation == "grouped_mm"
    assert len(tuple(configured.parameters())) == len(parameters)
    for actual, original in zip(configured.parameters(), parameters):
        assert actual is original
        assert actual.device.type == "cpu"

    logits = torch.randn(16, configured.config.num_experts, requires_grad=True)
    loss = configured.load_balancing_loss_func(
        (logits,),
        num_experts=configured.config.num_experts,
        top_k=configured.config.num_experts_per_tok,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
