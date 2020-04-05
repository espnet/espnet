import torch

from espnet2.optimizers.sgd import SGD


def test_SGD():
    linear = torch.nn.Linear(1, 1)
    opt = SGD(linear.parameters())
    x = torch.randn(1, 1)
    linear(x).sum().backward()
    opt.step()
