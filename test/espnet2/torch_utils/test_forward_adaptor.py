import pytest
import torch

from espnet2.torch_utils.forward_adaptor import ForwardAdaptor


class Model(torch.nn.Module):
    def func(self, x):
        return x


def test_ForwardAdaptor():
    model = Model()
    x = torch.randn(2, 2)
    assert (ForwardAdaptor(model, "func")(x) == x).all()


def test_ForwardAdaptor_no_func():
    model = Model()
    with pytest.raises(ValueError):
        ForwardAdaptor(model, "aa")
