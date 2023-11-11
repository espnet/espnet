#!/usr/bin/env python3
import pytest
import torch

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, trim_by_ctc_posterior

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


def test_trim_by_ctc_posterior():
    # eg1: ctc: 3 frames + 5 frame tolerance; mask: 10 frames -> 8 frames
    # eg2: ctc: 7 frames + 5 frame tolearnce; mask: 4  frames -> 4 frames
    h = torch.randn(2, 10, 7)
    ctc_prob = torch.tensor(
        [
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
        ]
    )
    masks = (
        torch.Tensor(
            [
                [True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, False, False, False, False, False, False],
            ]
        )
        .unsqueeze(1)
        .bool()
    )

    # PositionalEncoding
    pos_emb = torch.randn(2, 10, 7)
    h_hat, masks_hat, pos_emb_hat = trim_by_ctc_posterior(h, ctc_prob, masks, pos_emb)
    assert torch.all(torch.eq(h_hat, h[:, :8]))
    assert torch.all(torch.eq(masks_hat, masks[:, :, :8]))
    assert torch.all(torch.eq(pos_emb_hat, pos_emb_hat[:, :8]))

    # RelPositionalEncoding
    pos_emb = torch.randn(2, 19, 7)
    h_hat, masks_hat, pos_emb_hat = trim_by_ctc_posterior(h, ctc_prob, masks, pos_emb)
    assert torch.all(torch.eq(h_hat, h[:, :8]))
    assert torch.all(torch.eq(masks_hat, masks[:, :, :8]))
    assert torch.all(torch.eq(pos_emb_hat, pos_emb[:, 2:17]))
