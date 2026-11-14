"""Unit tests for espnet3.systems.lace.src.compression."""

import pytest
import torch

from espnet3.systems.lace.src.compression import (
    BaseCompressionModel,
    CompressionOutput,
)


class _PerFrameCompression(BaseCompressionModel):
    """Smallest concrete subclass: every frame is its own segment."""

    def forward(self, x, rate=None, padding_mask=None, **kwargs):
        """Return the shared per-frame segmentation."""
        return self._build_no_rate_output(x, padding_mask)


def test_base_class_is_abstract():
    """``BaseCompressionModel`` cannot be instantiated without ``forward``."""
    with pytest.raises(TypeError):
        BaseCompressionModel()


def test_no_rate_output_is_per_frame_identity():
    """With no rate every frame is a segment and the features pass through."""
    x = torch.randn(2, 5, 3)
    output = _PerFrameCompression()(x)
    assert isinstance(output, CompressionOutput)
    assert torch.equal(output.reconstructed_features, x)
    assert torch.equal(output.segment_idx, torch.arange(5).expand(2, 5))
    assert output.boundary_soft[:, 0].eq(0).all()
    assert output.boundary_soft[:, 1:].eq(1).all()
    assert torch.equal(output.expected_length, torch.full((2, 1), 5.0))


def test_no_rate_output_masks_padding():
    """Padded positions carry no boundary and join the last valid segment."""
    x = torch.randn(1, 5, 3)
    padding_mask = torch.tensor([[False, False, False, True, True]])
    output = _PerFrameCompression()(x, padding_mask=padding_mask)
    assert output.boundary_soft.tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0]]
    assert output.segment_idx.tolist() == [[0, 1, 2, 2, 2]]
    assert output.expected_length.tolist() == [[3.0]]


def test_segment_average_returns_segment_means_at_frame_resolution():
    """Each frame is replaced by the mean of its segment."""
    x = torch.tensor([[[1.0], [3.0], [10.0], [20.0], [30.0]]])
    segment_idx = torch.tensor([[0, 0, 1, 1, 1]])
    averaged = _PerFrameCompression()._average_segments(x, segment_idx, None)
    expected = torch.tensor([[[2.0], [2.0], [20.0], [20.0], [20.0]]])
    assert averaged.shape == x.shape
    assert torch.allclose(averaged, expected, atol=1e-4)


def test_segment_average_ignores_padded_frames():
    """Padded frames neither contribute to nor skew a segment mean."""
    x = torch.tensor([[[1.0], [3.0], [100.0]]])
    segment_idx = torch.tensor([[0, 0, 0]])
    padding_mask = torch.tensor([[False, False, True]])
    averaged = _PerFrameCompression()._average_segments(x, segment_idx, padding_mask)
    assert torch.allclose(averaged, torch.full((1, 3, 1), 2.0), atol=1e-4)


def test_segment_average_keeps_gradient():
    """The scatter/gather averaging stays differentiable."""
    x = torch.randn(2, 6, 4, requires_grad=True)
    segment_idx = torch.tensor([[0, 0, 1, 1, 2, 2]]).expand(2, 6)
    _PerFrameCompression()._average_segments(x, segment_idx, None).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
