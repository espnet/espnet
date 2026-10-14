"""Unit tests for espnet3.systems.lace.dp_segmentation."""

import pytest
import torch

from espnet3.systems.lace.dp_segmentation import DPSegmentationCompression


def test_rate_none_returns_per_frame_segmentation():
    x = torch.randn(2, 7, 4)
    output = DPSegmentationCompression()(x, rate=None)
    assert torch.equal(output.reconstructed_features, x)


@pytest.mark.parametrize("rate, num_segments", [(0.5, 25), (1.0, 50), (0.25, 13)])
def test_segment_count_follows_rate(rate, num_segments):
    torch.manual_seed(0)
    x = torch.randn(2, 50, 8)
    output = DPSegmentationCompression(max_span=4)(x, rate=rate)
    assert output.expected_length.tolist() == [[num_segments]] * 2


def test_max_span_raises_segment_count_to_stay_feasible():
    torch.manual_seed(0)
    x = torch.randn(1, 12, 4)
    output = DPSegmentationCompression(max_span=2)(x, rate=0.1)
    assert output.expected_length.item() == 6.0
    assert torch.bincount(output.segment_idx[0]).max() <= 2


def test_dp_finds_the_minimum_distortion_split():
    values = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
    x = torch.tensor(values).view(1, 6, 1)
    # rate * (T - 1) = 0.3 * 5 -> k = 1 -> 2 segments
    output = DPSegmentationCompression(max_span=None)(x, rate=0.3)
    assert output.segment_idx.tolist() == [[0, 0, 0, 1, 1, 1]]
    assert torch.allclose(output.reconstructed_features, x, atol=1e-4)


def test_anchor_boundary_restricts_boundaries_to_anchor_set():
    torch.manual_seed(0)
    x = torch.randn(1, 12, 4)
    anchor_boundary = torch.zeros(1, 12, dtype=torch.long)
    anchor_boundary[0, [3, 6, 9]] = 1
    output = DPSegmentationCompression(max_span=4)(
        x, rate=0.3, anchor_boundary=anchor_boundary
    )
    assert (output.boundary_soft.bool() & ~anchor_boundary.bool()).sum() == 0
    assert output.expected_length.item() == 4.0


def test_empty_anchor_set_falls_back_to_unconstrained_dp():
    torch.manual_seed(0)
    x = torch.randn(1, 10, 4)
    model = DPSegmentationCompression()
    anchored = model(x, rate=0.5, anchor_boundary=torch.zeros(1, 10))
    free = model(x, rate=0.5)
    assert torch.equal(anchored.segment_idx, free.segment_idx)


def test_infeasible_anchor_set_falls_back_to_uniform_segments():
    torch.manual_seed(0)
    x = torch.randn(1, 12, 4)
    anchor_boundary = torch.zeros(1, 12, dtype=torch.long)
    anchor_boundary[0, 6] = 1  # two 6-frame segments exceed max_span=4
    output = DPSegmentationCompression(max_span=4)(
        x, rate=0.3, anchor_boundary=anchor_boundary
    )
    assert output.segment_idx.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]]


def test_padding_and_per_sample_rates():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 4)
    padding_mask = torch.zeros(2, 10, dtype=torch.bool)
    padding_mask[0, 6:] = True
    padding_mask[1, 1:] = True  # a single valid frame
    output = DPSegmentationCompression()(
        x, rate=torch.tensor([0.5, 0.5]), padding_mask=padding_mask
    )
    assert output.boundary_soft[0, 6:].eq(0).all()
    assert output.expected_length[1].item() == 1.0


def test_uniform_segment_fallback_covers_all_segments():
    segment_idx = DPSegmentationCompression._uniform_segment(7, 3, "cpu")
    assert segment_idx.tolist() == [0, 0, 0, 1, 1, 2, 2]


def test_gradient_flows_through_segment_average():
    x = torch.randn(1, 8, 4, requires_grad=True)
    output = DPSegmentationCompression()(x, rate=0.5)
    output.reconstructed_features.sum().backward()
    assert x.grad is not None
