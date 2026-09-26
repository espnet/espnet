import pytest
import torch

from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    _conv_out_length,
    _receptive_field_and_stride,
)

CLASSES = [
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
]


class _CountingConv(torch.nn.Module):
    """Wrap ``module.conv`` to record the time length it was given."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.seen = []

    def forward(self, x):
        self.seen.append(x.size(2))
        return self.conv(x)


def _module(cls):
    torch.manual_seed(0)
    module = cls(idim=20, odim=8, dropout_rate=0.0).eval()
    module.conv = _CountingConv(module.conv)
    return module


def _padded_input(tails, time=97, idim=20):
    """Random frames, then one constant frame repeated for the last ``tails[b]``."""
    torch.manual_seed(1)
    x = torch.randn(len(tails), time, idim)
    for b, tail in enumerate(tails):
        if tail > 0:
            x[b, time - tail :] = torch.randn(idim)
    return x


def _reference(module, x, mask):
    module.skip_constant_tail = False
    try:
        return module(x, mask)
    finally:
        module.skip_constant_tail = True


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("with_mask", [True, False])
@torch.no_grad()
def test_constant_tail_gives_full_result(cls, with_mask):
    module = _module(cls)
    x = _padded_input([70, 55, 40])  # long enough for every stride
    mask = torch.ones(3, 1, x.size(1), dtype=torch.bool) if with_mask else None
    if mask is not None:
        mask[2, :, 90:] = False
    expected, expected_mask = _reference(module, x, mask)
    module.conv.seen.clear()
    got, got_mask = module(x, mask)
    torch.testing.assert_close(got, expected)
    if with_mask:
        assert torch.equal(got_mask, expected_mask)
    else:
        assert got_mask is None
    # the convolutions ran on a prefix only
    assert module.conv.seen == [module.conv.seen[0]] and module.conv.seen[0] < x.size(1)


@pytest.mark.parametrize("cls", CLASSES)
@torch.no_grad()
def test_no_constant_tail_or_training_runs_full(cls):
    module = _module(cls)
    x = _padded_input([0, 0])
    module.conv.seen.clear()
    module(x, None)
    assert module.conv.seen == [x.size(1)]
    # one utterance without padding disables the shortcut for the batch
    x = _padded_input([40, 0])
    module.conv.seen.clear()
    module(x, None)
    assert module.conv.seen == [x.size(1)]
    # autograd enabled (gradients wanted in eval mode) never takes the shortcut
    with torch.enable_grad():
        module.conv.seen.clear()
        module(_padded_input([40, 40]), None)
        assert module.conv.seen == [x.size(1)]
    # training mode never takes the shortcut
    x = _padded_input([40, 40])
    module.train()
    module.conv.seen.clear()
    module(x, None)
    assert module.conv.seen == [x.size(1)]


@pytest.mark.parametrize("cls", CLASSES)
@torch.no_grad()
def test_all_constant_input(cls):
    module = _module(cls)
    x = torch.randn(1, 1, 20).expand(2, 97, 20).contiguous()
    expected, _ = _reference(module, x, None)
    got, _ = module(x, None)
    torch.testing.assert_close(got, expected)
    assert got.size(1) == expected.size(1)


def test_receptive_field_and_stride():
    assert _receptive_field_and_stride(((3, 2), (3, 2), (3, 2))) == (15, 8)
    assert _receptive_field_and_stride(((3, 2), (5, 3))) == (11, 6)
    assert _receptive_field_and_stride(((3, 1), (3, 1))) == (5, 1)
    # the crop keeps at least one output frame whose receptive field is pure padding
    convs = ((3, 2), (3, 2), (3, 2))
    rf, stride = _receptive_field_and_stride(convs)
    for first_pad in range(0, 60):
        t_crop = first_pad + rf + stride
        t = t_crop
        for k, s in convs:
            t = _conv_out_length(t, k, s)
        assert (t - 1) * stride >= first_pad
