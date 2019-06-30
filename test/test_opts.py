import argparse

import pytest
import torch

from espnet.opts.chainer_backend.adadelta import AdaDelta as AdaDelta_ch
from espnet.opts.chainer_backend.adagrad import AdaGrad as AdaGrad_ch
from espnet.opts.chainer_backend.adam import Adam as Adam_ch
from espnet.opts.pytorch_backend.adadelta import Adadelta as Adadelta_th
from espnet.opts.pytorch_backend.adagrad import Adagrad as Adagrad_th
from espnet.opts.pytorch_backend.adam import Adam as Adam_th
from espnet.opts.pytorch_backend.noam import NoamAdam as NoamAdam_th


@pytest.mark.parametrize('opt_class', [AdaDelta_ch, AdaGrad_ch, Adam_ch])
def test_chainer_opts(opt_class):
    parser = argparse.ArgumentParser()
    opt_class.add_arguments(parser)
    args = parser.parse_args([])
    opt_class.get(args)


@pytest.mark.parametrize('opt_class', [Adadelta_th, Adagrad_th, Adam_th,
                                       NoamAdam_th])
def test_pytorch_opts(opt_class):
    parser = argparse.ArgumentParser()
    opt_class.add_arguments(parser)
    args = parser.parse_args([])
    args.adim = 100

    net = torch.nn.Linear(1, 1)
    opt_class.get(net.parameters(), args)
