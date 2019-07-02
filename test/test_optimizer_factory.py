import argparse

import pytest
import torch

from espnet.opts.chainer_backend.adadelta import AdaDeltaFactory as AdaDelta_ch
from espnet.opts.chainer_backend.adagrad import AdaGradFactory as AdaGrad_ch
from espnet.opts.chainer_backend.adam import AdamFactory as Adam_ch
from espnet.opts.pytorch_backend.adadelta import AdadeltaFactory as Adadelta_th
from espnet.opts.pytorch_backend.adagrad import AdagradFactory as Adagrad_th
from espnet.opts.pytorch_backend.adam import AdamFactory as Adam_th
from espnet.opts.pytorch_backend.noam import NoamAdamFactory as NoamAdam_th


@pytest.mark.parametrize('opt_factory', [AdaDelta_ch, AdaGrad_ch, Adam_ch])
def test_chainer_opts(opt_factory):
    parser = argparse.ArgumentParser()
    opt_factory.add_arguments(parser)
    args = parser.parse_args([])
    opt_factory.create(args)


@pytest.mark.parametrize('opt_factory', [Adadelta_th, Adagrad_th, Adam_th,
                                         NoamAdam_th])
def test_pytorch_opts(opt_factory):
    parser = argparse.ArgumentParser()
    opt_factory.add_arguments(parser)
    args = parser.parse_args([])
    args.adim = 100

    net = torch.nn.Linear(1, 1)
    opt_factory.create(net.parameters(), args)
