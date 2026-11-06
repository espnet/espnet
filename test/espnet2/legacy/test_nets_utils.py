#!/usr/bin/env python3
import pytest
import torch

from espnet2.legacy.nets.pytorch_backend.nets_utils import (
    make_pad_mask,
)

test_cases = [
    # lengths, xs, length_dim, maxlen
    {"lengths": [3]},
    {"lengths": [3], "length_dim": 1},
    {"lengths": [3], "length_dim": 2},
    {"lengths": [3], "length_dim": 1, "maxlen": 3},
    {"lengths": [3], "length_dim": 1, "maxlen": 5},
    {"lengths": [5, 2, 3]},
    {"lengths": [5, 2, 3], "length_dim": 1},
    {"lengths": [5, 2, 3], "length_dim": 2},
    {"lengths": [5, 2, 3], "maxlen": 5},
    {"lengths": [5, 2, 3], "maxlen": 8},
    {"lengths": [5, 2, 3], "xs": torch.ones(3, 2, 4)},
    {"lengths": [5, 2, 3], "xs": torch.ones(3, 2, 4), "length_dim": 1},
    {"lengths": [5, 2, 3], "xs": torch.ones(3, 6, 6), "length_dim": 1},
    {"lengths": [5, 2, 3], "xs": torch.ones(3, 6, 6), "length_dim": 2},
    {"lengths": [5, 0, 3]},
    {"lengths": [5, 0, 3], "maxlen": 8},
    {"lengths": [5, 0, 3], "xs": torch.ones(3, 6, 6), "length_dim": 1},
    {"lengths": [0, 0]},
]


@pytest.mark.parametrize("test_case", test_cases)
def test_make_pad_mask(test_case):
    """Test if onnx-convertible make_pad_mask works correctly."""
    _tc = test_case.copy()
    lengths = _tc.pop("lengths")
    non_traceable_result = make_pad_mask(lengths, **_tc)
    traceable_result = make_pad_mask(torch.LongTensor(lengths), **_tc)
    assert (traceable_result == non_traceable_result).all()


@pytest.mark.parametrize("test_case", test_cases)
def test_trace_make_pad_mask(test_case):
    """Test if onnx-convertible make_pad_mask can be traced with torch.jit.trace

    If it's traceable then it can be exported to ONNX.
    """
    args, input_names, kwargs_trace, kwargs_non_trace = get_args(test_case.copy())
    mpm = MakePadMaskTest(input_names)
    traced_function = torch.jit.trace(mpm, args)
    traced_output = traced_function(torch.LongTensor([3, 1, 2]), **kwargs_trace)
    non_traced_output = make_pad_mask([3, 1, 2], **kwargs_non_trace)
    assert (traced_output == non_traced_output).all()


class MakePadMaskTest(torch.nn.Module):
    def __init__(self, input_names):
        super(MakePadMaskTest, self).__init__()
        self.input_names = input_names

    def forward(self, *args):
        if self.input_names == ("lengths",):
            return self.forward_lengths(*args)
        elif self.input_names == ("lengths", "length_dim"):
            return self.forward_dim(*args)
        elif self.input_names == ("lengths", "maxlen"):
            return self.forward_max(*args)
        elif self.input_names == ("lengths", "length_dim", "maxlen"):
            return self.forward_dim_max(*args)
        elif self.input_names == ("lengths", "xs"):
            return self.forward_xs(*args)
        elif self.input_names == ("lengths", "xs", "length_dim"):
            return self.forward_xs_dim(*args)

    def forward_lengths(self, lengths):
        return make_pad_mask(lengths)

    def forward_dim(self, lengths, length_dim):
        return make_pad_mask(lengths, length_dim=length_dim)

    def forward_max(self, lengths, maxlen):
        return make_pad_mask(lengths, maxlen=maxlen)

    def forward_dim_max(self, lengths, length_dim, maxlen):
        return make_pad_mask(lengths, length_dim=length_dim, maxlen=maxlen)

    def forward_xs(self, lengths, xs):
        return make_pad_mask(lengths, xs=xs)

    def forward_xs_dim(self, lengths, xs, length_dim):
        return make_pad_mask(lengths, xs=xs, length_dim=length_dim)


def get_args(tc):
    args = []
    input_names = []
    kwargs_trace = {}
    kwargs_non = {}
    lengths = tc.pop("lengths")
    args.append(torch.LongTensor(lengths))
    input_names.append("lengths")

    xs = tc.pop("xs", None)
    if xs is not None:
        args.append(xs)
        input_names.append("xs")
        kwargs_trace["xs"] = xs
        kwargs_non["xs"] = xs

    ld = tc.pop("length_dim", None)
    if ld is not None:
        args.append(torch.LongTensor([ld]))
        input_names.append("length_dim")
        kwargs_trace["length_dim"] = torch.LongTensor([ld])
        kwargs_non["length_dim"] = ld

    ml = tc.pop("maxlen", None)
    if ml is not None:
        args.append(torch.LongTensor([ml]))
        input_names.append("maxlen")
        kwargs_trace["maxlen"] = torch.LongTensor([ml])
        kwargs_non["maxlen"] = ml

    return tuple(args), tuple(input_names), kwargs_trace, kwargs_non


def test_make_pad_mask_long_input_is_linear_in_memory():
    """An hour-long input (~10^6 frames) must not build a (maxlen, maxlen) table.

    With the quadratic implementation this needed ~10^12 bytes.
    """
    maxlen = 1_000_000
    lengths = torch.LongTensor([maxlen, 3])
    xs = torch.zeros(2, maxlen, 1)
    mask = make_pad_mask(lengths, xs, length_dim=1)
    assert mask.shape == (2, maxlen, 1)
    assert not mask[0].any()
    assert not mask[1, :3].any() and mask[1, 3:].all()


def test_trace_make_pad_mask_with_another_maxlen():
    """A graph traced with a tensor maxlen follows a different maxlen later."""
    traced = torch.jit.trace(
        MakePadMaskTest(("lengths", "maxlen")),
        (torch.LongTensor([3, 1, 2]), torch.LongTensor([5])),
    )
    for maxlen in [3, 7]:
        out = traced(torch.LongTensor([3, 1, 2]), torch.LongTensor([maxlen]))
        assert torch.equal(out, make_pad_mask([3, 1, 2], maxlen=maxlen))


_accelerators = [
    d
    for d, ok in [
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    ]
    if ok
]


@pytest.mark.skipif(not _accelerators, reason="needs a CUDA or MPS device")
@pytest.mark.parametrize("device", _accelerators)
def test_make_pad_mask_cpu_lengths_with_xs_on_device(device):
    """lengths on the CPU and xs on an accelerator give the mask on xs's device."""
    lengths = torch.LongTensor([3, 1, 2])
    for xs, length_dim in [
        (torch.zeros(3, 4), -1),
        (torch.zeros(3, 2, 4), -1),
        (torch.zeros(3, 4, 2), 1),
    ]:
        mask = make_pad_mask(lengths, xs.to(device), length_dim=length_dim)
        assert mask.device.type == device
        assert torch.equal(mask.cpu(), make_pad_mask(lengths, xs, length_dim))
