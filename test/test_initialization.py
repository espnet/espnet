# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse

import numpy
import os
import pytest
import random


args = argparse.Namespace(
    elayers=4,
    subsample="1_2_2_1_1",
    etype="vggblstmp",
    eunits=320,
    eprojs=320,
    dlayers=2,
    dunits=300,
    atype="location",
    aconv_chans=10,
    aconv_filts=100,
    mtlalpha=0.5,
    lsm_type="",
    lsm_weight=0.0,
    sampling_probability=0.0,
    adim=320,
    dropout_rate=0.0,
    beam_size=3,
    penalty=0.5,
    maxlenratio=1.0,
    minlenratio=0.0,
    ctc_weight=0.2,
    verbose=True,
    char_list=[u"あ", u"い", u"う", u"え", u"お"],
    outdir=None,
    seed=1
)


def test_lecun_init_torch():
    torch = pytest.importorskip("torch")
    nseed = args.seed
    random.seed(nseed)
    torch.manual_seed(nseed)
    numpy.random.seed(nseed)
    os.environ["CHAINER_SEED"] = str(nseed)
    import espnet.nets.pytorch.e2e_asr_th as m
    model = m.Loss(m.E2E(40, 5, args), 0.5)
    b = model.predictor.ctc.ctc_lo.bias.data.numpy()
    assert numpy.all(b == 0.0)
    w = model.predictor.ctc.ctc_lo.weight.data.numpy()
    numpy.testing.assert_allclose(w.mean(), 0.0, 1e-2, 1e-2)
    numpy.testing.assert_allclose(w.var(), 1.0 / w.shape[1], 1e-2, 1e-2)

    for name, p in model.named_parameters():
        print(name)
        data = p.data.numpy()
        if "embed" in name:
            numpy.testing.assert_allclose(data.mean(), 0.0, 5e-2, 5e-2)
            numpy.testing.assert_allclose(data.var(), 1.0, 5e-2, 5e-2)
        elif "predictor.dec.decoder.0.bias_ih" in name:
            assert data.sum() == data.size // 4
        elif "predictor.dec.decoder.1.bias_ih" in name:
            assert data.sum() == data.size // 4
        elif data.ndim == 1:
            assert numpy.all(data == 0.0)
        else:
            numpy.testing.assert_allclose(data.mean(), 0.0, 5e-2, 5e-2)
            numpy.testing.assert_allclose(
                data.var(), 1.0 / numpy.prod(data.shape[1:]), 5e-2, 5e-2)


def test_lecun_init_chainer():
    nseed = args.seed
    random.seed(nseed)
    numpy.random.seed(nseed)
    os.environ["CHAINER_SEED"] = str(nseed)
    import espnet.nets.chainer.e2e_asr as m
    model = m.Loss(m.E2E(40, 5, args), 0.5)
    b = model.predictor.ctc.ctc_lo.b.data
    assert numpy.all(b == 0.0)
    w = model.predictor.ctc.ctc_lo.W.data
    numpy.testing.assert_allclose(w.mean(), 0.0, 1e-2, 1e-2)
    numpy.testing.assert_allclose(w.var(), 1.0 / w.shape[1], 1e-2, 1e-2)

    for name, p in model.namedparams():
        print(name)
        data = p.data
        if "lstm0/upward/b" in name:
            assert data.sum() == data.size // 4
        elif "lstm1/upward/b" in name:
            assert data.sum() == data.size // 4
        elif "embed" in name:
            numpy.testing.assert_allclose(data.mean(), 0.0, 5e-2, 5e-2)
            numpy.testing.assert_allclose(data.var(), 1.0, 5e-2, 5e-2)
        elif data.ndim == 1:
            assert numpy.all(data == 0.0)
        else:
            numpy.testing.assert_allclose(data.mean(), 0.0, 5e-2, 5e-2)
            numpy.testing.assert_allclose(
                data.var(), 1.0 / numpy.prod(data.shape[1:]), 5e-2, 5e-2)
