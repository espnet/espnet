"""Unit tests for espnet3.systems.lace.src.identity."""

import pytest
import torch

from espnet3.systems.lace.src.identity import IdentityCompression


@pytest.mark.parametrize("rate", [None, 0.1, 1.0])
def test_identity_ignores_rate_and_anchor(rate):
    """Rate and anchors do not change the per-frame output."""
    x = torch.randn(2, 6, 4)
    anchor_boundary = torch.zeros(2, 6, dtype=torch.long)
    output = IdentityCompression()(x, rate=rate, anchor_boundary=anchor_boundary)
    assert torch.equal(output.reconstructed_features, x)
    assert torch.equal(output.expected_length, torch.full((2, 1), 6.0))


def test_identity_respects_padding_mask():
    """Padded frames do not count as segments."""
    x = torch.randn(1, 4, 2)
    padding_mask = torch.tensor([[False, False, True, True]])
    output = IdentityCompression()(x, padding_mask=padding_mask)
    assert output.expected_length.tolist() == [[2.0]]
