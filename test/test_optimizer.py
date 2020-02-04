# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

import chainer
import numpy
import pytest
import torch

from espnet.optimizer.factory import dynamic_import_optimizer
from espnet.optimizer.parser import OPTIMIZER_PARSER_DICT


class ChModel(chainer.Chain):
    def __init__(self):
        super(ChModel, self).__init__()
        with self.init_scope():
            self.a = chainer.links.Linear(3, 1)

    def __call__(self, x):
        return chainer.functions.sum(self.a(x))


class ThModel(torch.nn.Module):
    def __init__(self):
        super(ThModel, self).__init__()
        self.a = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.a(x).sum()


def test_optimizer_parser():
    parser = argparse.ArgumentParser()
    opt_class = dynamic_import_optimizer("sgd", "pytorch")
    opt_class.add_arguments(parser)
    args = parser.parse_args(["--lr", "1.0"])
    assert args.lr == 1.0
    model = ThModel()
    opt = opt_class(model.parameters(), args)
    assert opt.param_groups[0]["lr"] == 1.0


@pytest.mark.parametrize("name", OPTIMIZER_PARSER_DICT.keys())
def test_optimizer_backend_compatible(name):
    torch.set_grad_enabled(True)
    # model construction
    ch_model = ChModel()
    th_model = ThModel()

    # copy params
    th_model.a.weight.data = torch.from_numpy(numpy.copy(ch_model.a.W.data))
    th_model.a.bias.data = torch.from_numpy(numpy.copy(ch_model.a.b.data))

    # optimizer setup
    th_opt = dynamic_import_optimizer(name, "pytorch").build(th_model.parameters())
    ch_opt = dynamic_import_optimizer(name, "chainer").build(ch_model)

    # forward
    ch_model.cleargrads()
    data = numpy.random.randn(2, 3).astype(numpy.float32)
    ch_loss = ch_model(data)
    th_loss = th_model(torch.from_numpy(data))
    chainer.functions.sum(ch_loss).backward()
    th_loss.backward()
    numpy.testing.assert_allclose(ch_loss.data, th_loss.item(), rtol=1e-6)
    ch_opt.update()
    th_opt.step()
    numpy.testing.assert_allclose(
        ch_model.a.W.data, th_model.a.weight.data.numpy(), rtol=1e-6)
    numpy.testing.assert_allclose(
        ch_model.a.b.data, th_model.a.bias.data.numpy(), rtol=1e-6)
