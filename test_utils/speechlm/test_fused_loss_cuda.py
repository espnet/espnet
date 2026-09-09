# Copyright 2026 Carnegie Mellon University
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Compare SpeechLM's real fused CUDA loss and gradients with PyTorch."""

import pytest
import torch
import torch.nn.functional as F

from espnet2.speechlm.model.speechlm.lm.loss import fused_cross_entropy_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("streams", [1, 8])
@pytest.mark.parametrize("z_weight", [0.0, 1e-5])
def test_fused_loss_matches_cross_entropy(streams, z_weight):
    torch.manual_seed(7)
    hidden = torch.randn(2, 13, streams, 32, device="cuda", requires_grad=True)
    weight = torch.randn(96, 32, device="cuda", requires_grad=True)
    targets = torch.randint(1, 96, (2, 13, streams), device="cuda")
    if streams > 1:
        targets[:, :, 1:] = torch.randint(64, 96, (2, 13, streams - 1), device="cuda")
    mask = torch.ones_like(targets, dtype=torch.float32)
    mask[:, :3] = 0
    class_weight = torch.ones(96, device="cuda")
    class_weight[64:] = 1 / streams

    actual, count, stats = fused_cross_entropy_loss(
        hidden,
        targets,
        mask,
        weight,
        (64, 96) if streams > 1 else None,
        streams,
        True,
        z_loss_weight=z_weight,
        ce_weight=class_weight,
    )
    expected = hidden.new_zeros(())
    for stream_range, start, end in [(slice(0, 1), 0, 96)] + (
        [(slice(1, None), 64, 96)] if streams > 1 else []
    ):
        states = hidden[:, :-1, stream_range].reshape(-1, 32)
        ids = targets[:, 1:, stream_range].reshape(-1) - start
        active = mask[:, 1:, stream_range].reshape(-1).bool()
        logits = F.linear(states, weight[start:end]).float()
        expected = expected + F.cross_entropy(
            logits[active],
            ids[active],
            weight=class_weight[start:end],
            reduction="sum",
        )
        expected = expected + z_weight * logits[active].logsumexp(-1).square().sum()

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-3)
    torch.testing.assert_close(count, mask[:, 1:, 0].sum())
    for value in stats.values():
        assert torch.isfinite(value).all()
    actual_grads = torch.autograd.grad(actual, (hidden, weight), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (hidden, weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-4, atol=2e-4)
