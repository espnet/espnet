"""Unit tests for espnet3.systems.lace.src.cosine_similarity."""

import torch

from espnet3.systems.lace.src.cosine_similarity import CosineSimilarityCompression


def make_two_segment_input():
    """Three frames along one axis, then three along an orthogonal axis."""
    first = torch.tensor([1.0, 0.0]).expand(3, 2)
    second = torch.tensor([0.0, 1.0]).expand(3, 2)
    return torch.cat([first, second]).unsqueeze(0)  # (1, 6, 2)


def test_rate_none_returns_per_frame_segmentation():
    x = torch.randn(2, 7, 4)
    output = CosineSimilarityCompression()(x, rate=None)
    assert torch.equal(output.reconstructed_features, x)
    assert torch.equal(output.expected_length, torch.full((2, 1), 7.0))


def test_boundary_placed_where_similarity_drops_below_rate():
    x = make_two_segment_input()
    output = CosineSimilarityCompression(max_tokens_per_group=None)(x, rate=0.5)
    assert output.boundary_soft.tolist() == [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    assert output.segment_idx.tolist() == [[0, 0, 0, 1, 1, 1]]
    assert output.expected_length.tolist() == [[2.0]]
    assert torch.allclose(output.reconstructed_features, x, atol=1e-4)


def test_max_tokens_per_group_splits_long_segments():
    x = torch.ones(1, 7, 2)  # one natural segment of 7 identical frames
    output = CosineSimilarityCompression(max_tokens_per_group=3)(x, rate=0.5)
    assert output.segment_idx.tolist() == [[0, 0, 0, 1, 1, 1, 2]]


def test_higher_rate_never_gives_fewer_segments():
    torch.manual_seed(0)
    x = torch.randn(3, 30, 8)
    model = CosineSimilarityCompression(max_tokens_per_group=None)
    low = model(x, rate=0.1).expected_length
    high = model(x, rate=0.9).expected_length
    assert (high >= low).all()


def test_per_sample_rate_tensor_matches_scalar_rates():
    torch.manual_seed(0)
    x = torch.randn(2, 20, 8)
    model = CosineSimilarityCompression()
    output = model(x, rate=torch.tensor([0.2, 0.8]))
    for index, rate in enumerate([0.2, 0.8]):
        single = model(x[index : index + 1], rate=rate)
        assert torch.equal(output.segment_idx[index], single.segment_idx[0])


def test_anchor_boundary_restricts_boundaries_to_anchor_set():
    torch.manual_seed(0)
    x = torch.randn(2, 20, 8)
    anchor_boundary = torch.zeros(2, 20, dtype=torch.long)
    anchor_boundary[:, [5, 10, 15]] = 1
    model = CosineSimilarityCompression(max_tokens_per_group=None)
    output = model(x, rate=1.0, anchor_boundary=anchor_boundary)
    assert (output.boundary_soft.bool() & ~anchor_boundary.bool()).sum() == 0


def test_padding_positions_never_carry_boundaries():
    torch.manual_seed(0)
    x = torch.randn(1, 10, 4)
    padding_mask = torch.zeros(1, 10, dtype=torch.bool)
    padding_mask[0, 6:] = True
    output = CosineSimilarityCompression()(x, rate=1.0, padding_mask=padding_mask)
    assert output.boundary_soft[0, 6:].eq(0).all()
