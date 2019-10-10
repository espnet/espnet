from espnet.scheduler.chainer import ChainerScheduler
from espnet.scheduler.pytorch import PyTorchScheduler
from espnet.scheduler import scaler

import chainer
import numpy
import pytest
import torch


@pytest.mark.parametrize("name", scaler.SCALER_DICT.keys())
def test_scaler(name):
    s = scaler.dynamic_import_scaler(name).build("lr")
    assert s.key == "lr"
    assert isinstance(s.scale(0), float)
    assert isinstance(s.scale(1000), float)


def test_pytorch_scheduler():
    warmup = 30000
    s = scaler.NoamScaler.build("lr", warmup=warmup)
    net = torch.nn.Linear(2, 1)
    o = torch.optim.SGD(net.parameters(), lr=1.0)
    scheduler = PyTorchScheduler([s], o)
    scheduler.step(0)
    for g in o.param_groups:
        assert g["lr"] == s.scale(0)

    scheduler.step(warmup)
    for g in o.param_groups:
        numpy.testing.assert_allclose(g["lr"], 1.0, rtol=1e-4)


def test_chainer_scheduler():
    warmup = 30000
    s = scaler.NoamScaler.build("lr", warmup=warmup)
    net = chainer.links.Linear(2, 1)
    o = chainer.optimizers.SGD(lr=1.0)
    o.setup(net)
    scheduler = ChainerScheduler([s], o)
    scheduler.step(0)
    assert o.lr == s.scale(0)

    scheduler.step(warmup)
    numpy.testing.assert_allclose(o.lr, 1.0, rtol=1e-4)
