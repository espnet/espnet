"""Unit tests for espnet3.systems.lace.density_peak."""

import torch

from espnet3.systems.lace.density_peak import DensityPeakCompression


def test_rate_none_returns_per_frame_segmentation():
    x = torch.randn(2, 7, 4)
    output = DensityPeakCompression()(x, rate=None)
    assert torch.equal(output.reconstructed_features, x)


def test_output_shapes_and_segment_count():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 8)
    output = DensityPeakCompression()(x, rate=0.5)
    assert output.boundary_soft.shape == (2, 16)
    assert output.segment_idx.shape == (2, 16)
    assert output.reconstructed_features.shape == (2, 16, 8)
    assert torch.equal(
        output.expected_length.squeeze(-1), output.segment_idx.max(dim=1).values + 1.0
    )


def test_segments_never_exceed_max_span():
    torch.manual_seed(0)
    x = torch.ones(1, 12, 4) + 0.01 * torch.randn(1, 12, 4)
    output = DensityPeakCompression(max_span=3)(x, rate=0.0)
    assert torch.bincount(output.segment_idx[0]).max() <= 3


def test_identical_frames_merge_and_distinct_frames_split():
    first = torch.tensor([1.0, 0.0]).expand(2, 2)
    second = torch.tensor([-1.0, 0.0]).expand(2, 2)
    x = torch.cat([first, second]).unsqueeze(0)  # (1, 4, 2)
    output = DensityPeakCompression(k=1, beta=0.0, max_span=4)(x, rate=0.5)
    assert output.segment_idx.tolist() == [[0, 0, 1, 1]]


def test_higher_rate_never_gives_fewer_segments():
    torch.manual_seed(0)
    x = torch.randn(2, 20, 8)
    model = DensityPeakCompression()
    low = model(x, rate=0.1).expected_length
    high = model(x, rate=0.99).expected_length
    assert (high >= low).all()


def test_padding_and_single_frame_sequences():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 4)
    padding_mask = torch.zeros(2, 8, dtype=torch.bool)
    padding_mask[0, 5:] = True
    padding_mask[1, 1:] = True  # a single valid frame
    output = DensityPeakCompression()(x, rate=0.5, padding_mask=padding_mask)
    assert output.boundary_soft[0, 5:].eq(0).all()
    assert output.expected_length[1].item() == 1.0


def test_per_sample_rate_tensor_and_ignored_anchor():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 4)
    model = DensityPeakCompression()
    anchor_boundary = torch.zeros(2, 10, dtype=torch.long)
    output = model(x, rate=torch.tensor([0.3, 0.9]), anchor_boundary=anchor_boundary)
    for index, rate in enumerate([0.3, 0.9]):
        single = model(x[index : index + 1], rate=rate)
        assert torch.equal(output.segment_idx[index], single.segment_idx[0])
