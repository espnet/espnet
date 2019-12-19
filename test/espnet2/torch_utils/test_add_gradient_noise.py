import torch

from espnet2.torch_utils.add_gradient_noise import add_gradient_noise


def test_add_gradient_noise():
    linear = torch.nn.Linear(1, 1)
    linear(torch.rand(1, 1)).sum().backward()
    add_gradient_noise(linear, 100)
