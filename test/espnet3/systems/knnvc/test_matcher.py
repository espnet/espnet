"""Tests for the kNN matcher."""

import pytest
import torch
import torch.nn.functional as F

from espnet3.systems.knnvc.matcher import (
    compute_cosine_distances,
    match_features,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_cosine_distances_match_reference       | Equals 1 - cosine_similarity |
# | test_cosine_distances_validate_shapes       | Bad ranks / dims raise.      |
# | test_match_features_topk_one_returns_nearest| k=1 copies the nearest frame.|
# | test_match_features_averages_neighbours     | k=2 averages the two nearest.|
# | test_match_features_uses_synth_set          | Neighbours indexed into      |
# |                                             | synth_set when provided.     |
# | test_match_features_clamps_topk             | k > pool size is clamped.    |
# | test_match_features_validation              | Empty pool / bad k / bad     |
# |                                             | synth_set raise ValueError.  |


def test_cosine_distances_match_reference():
    torch.manual_seed(0)
    source = torch.randn(5, 7)
    pool = torch.randn(9, 7)
    expected = 1 - F.cosine_similarity(source[:, None], pool[None], dim=-1)
    torch.testing.assert_close(
        compute_cosine_distances(source, pool), expected, atol=1e-5, rtol=1e-5
    )


def test_cosine_distances_validate_shapes():
    with pytest.raises(ValueError, match="2-D"):
        compute_cosine_distances(torch.zeros(3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="dims differ"):
        compute_cosine_distances(torch.zeros(2, 3), torch.zeros(2, 4))


def test_match_features_topk_one_returns_nearest():
    pool = torch.eye(4)
    query = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.2, 0.8]])
    out = match_features(query, pool, topk=1)
    torch.testing.assert_close(out, torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 1.0]]))


def test_match_features_averages_neighbours():
    pool = torch.eye(4)
    query = torch.tensor([[0.7, 0.7, 0.0, 0.0]])
    out = match_features(query, pool, topk=2)
    torch.testing.assert_close(out, torch.tensor([[0.5, 0.5, 0.0, 0.0]]))


def test_match_features_uses_synth_set():
    pool = torch.eye(3)
    synth = torch.arange(9, dtype=torch.float32).view(3, 3)
    query = torch.tensor([[0.0, 1.0, 0.0]])
    out = match_features(query, pool, synth_set=synth, topk=1)
    torch.testing.assert_close(out, synth[1:2])


def test_match_features_clamps_topk():
    pool = torch.eye(2)
    query = torch.tensor([[1.0, 0.0]])
    out = match_features(query, pool, topk=10)
    torch.testing.assert_close(out, torch.tensor([[0.5, 0.5]]))


def test_match_features_validation():
    query = torch.ones(1, 2)
    with pytest.raises(ValueError, match="at least one frame"):
        match_features(query, torch.zeros(0, 2))
    with pytest.raises(ValueError, match="topk"):
        match_features(query, torch.ones(2, 2), topk=0)
    with pytest.raises(ValueError, match="aligned"):
        match_features(query, torch.ones(2, 2), synth_set=torch.ones(3, 2))
