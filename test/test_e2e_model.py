# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import importlib
import argparse

import pytest
import numpy
import chainer


def make_arg(etype):
    return argparse.Namespace(
    elayers = 4,
    subsample = "1_2_2_1_1",
    etype = "vggblstmp",
    eunits = 100,
    eprojs = 100,
    dlayers=1,
    dunits=300,
    atype="location",
    aconv_chans=10,
    aconv_filts=100,
    mtlalpha=0.5,
    adim=320,
    dropout_rate=0.0,
    beam_size=3,
    penalty=0.5,
    maxlenratio=1.0,
    minlenratio=0.0,
    verbose = 2,
    char_list = [u"あ", u"い", u"う", u"え", u"お"],
    outdir = None
    )


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_model_trainable_and_decodable(etype):
    args = make_arg(etype)
    for m_str in ["e2e_asr_attctc", "e2e_asr_attctc_th"]:
        try:
            import torch
        except:
            if m_str[-3:] == "_th":
                pytest.skip("pytorch is not installed")

        m = importlib.import_module(m_str)
        model = m.Loss(m.E2E(40, 5, args), 0.5)
        out_data = "1 2 3 4"
        data = [
            ("aaa", dict(feat=numpy.random.randn(100, 40).astype(numpy.float32), tokenid=out_data)),
            ("bbb", dict(feat=numpy.random.randn(200, 40).astype(numpy.float32), tokenid=out_data))
        ]
        attn_loss = model(data)
        attn_loss.backward() # trainable

        in_data = data[0][1]["feat"]
        y = model.predictor.recognize(in_data, args, args.char_list) # decodable



def init_torch_weight_const(m, val):
    for p in m.parameters():
        if p.dim() > 1:
            p.data.fill_(val)


def init_chainer_weight_const(m, val):
    for p in m.params():
        if p.data.ndim > 1:
            p.data[:] = val


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_loss_and_ctc_grad(etype):
    args = make_arg(etype)
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    try:
        import torch
    except:
        pytest.skip("pytorch is not installed")
    import e2e_asr_attctc as ch
    import e2e_asr_attctc_th as th
    ch_model = ch.E2E(40, 5, args)
    ch_model.cleargrads()
    th_model = th.E2E(40, 5, args)

    const = 1e-4
    init_torch_weight_const(th_model, const)
    init_chainer_weight_const(ch_model, const)

    out_data = "1 2 3 4"
    data = [
        ("aaa", dict(feat=numpy.random.randn(200, 40).astype(numpy.float32), tokenid=out_data)),
        ("bbb", dict(feat=numpy.random.randn(100, 40).astype(numpy.float32), tokenid=out_data)),
        ("cc", dict(feat=numpy.random.randn(100, 40).astype(numpy.float32), tokenid=out_data))
    ]

    ch_ctc, ch_att, ch_acc = ch_model(data)
    th_ctc, th_att, th_acc = th_model(data)

    # test masking
    ch_ench = ch_model.att.pre_compute_enc_h.data
    th_ench = th_model.att.pre_compute_enc_h.data.numpy()
    numpy.testing.assert_equal(ch_ench == 0.0, th_ench == 0.0)

    # test loss with constant weights (1.0) and bias (0.0) except for foget-bias (1.0)
    numpy.testing.assert_allclose(ch_ctc.data, th_ctc.data.numpy())
    numpy.testing.assert_allclose(ch_att.data, th_att.data.numpy())

    # test ctc grads
    ch_ctc.backward()
    th_ctc.backward()
    numpy.testing.assert_allclose(ch_model.ctc.ctc_lo.W.grad,
                                  th_model.ctc.ctc_lo.weight.grad.data.numpy(), 1e-7, 1e-8)
    numpy.testing.assert_allclose(ch_model.ctc.ctc_lo.b.grad,
                                  th_model.ctc.ctc_lo.bias.grad.data.numpy(), 1e-5, 1e-6)
    

    # test cross-entropy grads
    ch_model.cleargrads()
    th_model.zero_grad()

    ch_ctc, ch_att, ch_acc = ch_model(data)
    th_ctc, th_att, th_acc = th_model(data)
    ch_att.backward()
    th_att.backward()
    numpy.testing.assert_allclose(ch_model.dec.output.W.grad,
                                  th_model.dec.output.weight.grad.data.numpy(), 1e-7, 1e-8)
    numpy.testing.assert_allclose(ch_model.dec.output.b.grad,
                                  th_model.dec.output.bias.grad.data.numpy(), 1e-5, 1e-6)
    
    


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_zero_length_target(etype):
    args = make_arg(etype)
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    try:
        import torch
    except:
        pytest.skip("pytorch is not installed")
    import e2e_asr_attctc as ch
    import e2e_asr_attctc_th as th
    ch_model = ch.E2E(40, 5, args)
    ch_model.cleargrads()
    th_model = th.E2E(40, 5, args)

    out_data = ""
    data = [
        ("aaa", dict(feat=numpy.random.randn(200, 40).astype(numpy.float32), tokenid="1")),
        ("bbb", dict(feat=numpy.random.randn(100, 40).astype(numpy.float32), tokenid="")),
        ("cc", dict(feat=numpy.random.randn(100, 40).astype(numpy.float32), tokenid="1 2"))
    ]

    ch_ctc, ch_att, ch_acc = ch_model(data)
    th_ctc, th_att, th_acc = th_model(data)

    # NOTE: We ignore all zero length case because chainer also fails. Have a nice data-prep!
    # out_data = ""
    # data = [
    #     ("aaa", dict(feat=numpy.random.randn(200, 40).astype(numpy.float32), tokenid="")),
    #     ("bbb", dict(feat=numpy.random.randn(100, 40).astype(numpy.float32), tokenid="")),
    #     ("cc", dict(feat=numpy.random.randn(100, 40).astype(numpy.float32), tokenid=""))
    # ]
    # ch_ctc, ch_att, ch_acc = ch_model(data)
    # th_ctc, th_att, th_acc = th_model(data)
