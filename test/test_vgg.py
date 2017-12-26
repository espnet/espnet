# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import chainer
import numpy
import pytest

torch = pytest.importorskip("torch")

import e2e_asr_attctc as ch
import e2e_asr_attctc_th as th



def init_torch_weight_const(m, val):
    for p in m.parameters():
        if p.dim() > 1:
            p.data.fill_(val)
        else:
            p.data.fill_(0.0)


def init_chainer_weight_const(m, val):
    for p in m.params():
        if p.data.ndim > 1:
            p.data[:] = val
        else:
            p.data[:] = 0.0


def test_vgg():
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    ch_model = ch.VGG2L()
    ch_model.cleargrads()
    th_model = th.VGG2L()

    const = 1e-4
    init_torch_weight_const(th_model, const)
    init_chainer_weight_const(ch_model, const)

    out_data = "1 2 3 4"
    data = [
        ("aaa", dict(feat=numpy.random.randn(20, 40).astype(
            numpy.float32), tokenid=out_data)),
        ("bbb", dict(feat=numpy.random.randn(10, 40).astype(
            numpy.float32), tokenid=out_data)),
        ("cc", dict(feat=numpy.random.randn(10, 40).astype(
            numpy.float32), tokenid=out_data))
    ]

    xlen = [d[1]["feat"].shape[0] for d in data]
    ch_ylist, ch_ylen = ch_model([chainer.Variable(d[1]["feat"]) for d in data], xlen)
    xpad = th.pad_list([torch.autograd.Variable(torch.from_numpy(d[1]["feat"])) for d in data], 0.0)
    th_ypad, th_ylen = th_model(xpad, xlen)
    
    assert th_ylen.tolist() == ch_ylen.tolist()

    for i, l in enumerate(th_ylen):
        numpy.testing.assert_allclose(th_ypad[i, :l].data.numpy(), ch_ylist[i].data, 1e-6, 1e-6)

    th_ypad.sum().backward()
    sum([chainer.functions.sum(y) for y in ch_ylist]).backward()
    numpy.testing.assert_allclose(th_model.conv2_2.weight.grad.data.numpy(),
                                  ch_model.conv2_2.W.grad, 1e-6, 1e-6)
    numpy.testing.assert_allclose(th_model.conv1_1.weight.grad.data.numpy(),
                                  ch_model.conv1_1.W.grad, 1e-6, 1e-6)
