import numpy as np
import pytest
import torch
import torch.nn.functional as F

from espnet2.torch_utils.check_look_ahead import check_look_ahead, frame_last_sample

T = 40


def _conv(future):
    """(B, T) -> (B, T): a 3-tap filter reading frames t - 2 + future .. t + future."""
    weight = torch.tensor([[[0.5, -1.0, 2.0]]])

    def forward(x):
        y = F.pad(x.unsqueeze(1), (2 - future, future))
        return F.conv1d(y, weight).squeeze(1)

    return forward


@pytest.mark.parametrize("future", [0, 1, 3])
def test_a_filter_passes_at_its_look_ahead(future):
    x = torch.randn(2, T)
    check = check_look_ahead(_conv(future), x, lambda i: i + future, range(1, T))
    assert check.unread == 0.0
    assert check.read > 0.0


@pytest.mark.parametrize("future", [1, 3])
def test_a_filter_fails_below_its_look_ahead(future):
    with pytest.raises(AssertionError, match="may read only up to"):
        check_look_ahead(_conv(future), torch.randn(2, T), lambda i: i, range(1, T))


def test_tight_fails_above_the_look_ahead():
    forward, x = _conv(1), torch.randn(2, T)
    with pytest.raises(AssertionError, match="reads less far ahead"):
        check_look_ahead(forward, x, lambda i: i + 2, range(3, T))
    check = check_look_ahead(forward, x, lambda i: i + 2, range(3, T), tight=False)
    assert check.unread == 0.0


def test_atol_allows_small_changes():
    def forward(x):
        return x + 1e-7 * x.sum(dim=1, keepdim=True)  # every frame reads all

    x = torch.randn(1, T)
    with pytest.raises(AssertionError):
        check_look_ahead(forward, x, lambda i: i, [20])
    check_look_ahead(forward, x, lambda i: i, [20], atol=1e-3)


def test_numpy_output_and_another_time_dim():
    conv = _conv(0)

    def forward(x):  # (B, D, T) -> numpy (B, D, T)
        return np.stack([conv(x[:, d]).numpy() for d in range(x.size(1))], axis=1)

    check = check_look_ahead(forward, torch.randn(2, 3, T), lambda i: i, [5, 17], 2)
    assert check.unread == 0.0 and check.read > 0.0


def test_frame_last_sample():
    last = frame_last_sample(hop_length=128, win_length=512)
    assert last(0) == 511 and last(1) == 639
    assert frame_last_sample(128, 512, look_ahead=2)(0) == 2 * 128 + 511
    assert frame_last_sample(128, 512, center=True)(0) == 255


def test_cut_point_outside_the_input_raises():
    with pytest.raises(ValueError):
        check_look_ahead(_conv(0), torch.randn(1, T), lambda i: i, [T])
