import pytest
import torch

from espnet2.torch_utils.initialize import initialize

initialize_types = {}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 3)
        self.l1 = torch.nn.Linear(2, 2)
        self.rnn_cell = torch.nn.LSTMCell(2, 2)
        self.rnn = torch.nn.LSTM(2, 2)
        self.emb = torch.nn.Embedding(1, 1)
        self.norm = torch.nn.LayerNorm(1)


class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(2, 2, 3)


@pytest.mark.parametrize(
    "init",
    [
        "chainer",
        "xavier_uniform",
        "xavier_normal",
        "kaiming_normal",
        "kaiming_uniform",
        "dummy",
    ],
)
def test_initialize(init):
    model = Model()
    if init == "dummy":
        with pytest.raises(ValueError):
            initialize(model, init)
    else:
        initialize(model, init)


def test_5dim():
    model = Model2()
    with pytest.raises(NotImplementedError):
        initialize(model, "chainer")
