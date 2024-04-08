import pytest
import torch

from espnet2.asr.frontend.asteroid_frontend import AsteroidFrontend
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed


def test_frontend_repr():
    frontend = AsteroidFrontend()
    print(frontend)


def test_frontend_output_size():
    frontend = AsteroidFrontend(sinc_filters=16)
    assert frontend.output_size() == 16


@pytest.mark.parametrize("train", [True, False])
def test_frontend_backward(train):
    frontend = AsteroidFrontend(
        sinc_filters=16,
        sinc_kernel_size=128,
        sinc_stride=16,
    )
    if train:
        frontend.train()
    else:
        frontend.eval()
    set_all_random_seed(14)
    x = torch.randn(2, 1000, requires_grad=True)
    x_lengths = torch.LongTensor([1000, 980])
    y, y_lengths = frontend(x, x_lengths)
    y.sum().backward()
