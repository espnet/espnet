# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import importlib
import os
import tempfile

import chainer
import numpy as np
import pytest
import torch

from espnet.nets.e2e_asr_th import pad_list


def make_arg(**kwargs):
    defaults = dict(
        elayers=4,
        subsample="1_2_2_1_1",
        etype="vggblstm",
        eunits=100,
        eprojs=100,
        dlayers=1,
        dunits=300,
        atype="location",
        aheads=4,
        awin=5,
        aconv_chans=10,
        aconv_filts=100,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=320,
        dropout_rate=0.0,
        nbest=5,
        beam_size=3,
        penalty=0.5,
        maxlenratio=1.0,
        minlenratio=0.0,
        ctc_weight=0.2,
        verbose=2,
        char_list=[u"あ", u"い", u"う", u"え", u"お"],
        outdir=None,
        ctc_type="chainer"
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare_inputs(mode, ilens=[150, 100], olens=[4, 3], is_cuda=False):
    np.random.seed(1)
    assert len(ilens) == len(olens)
    xs = [np.random.randn(ilen, 40).astype(np.float32) for ilen in ilens]
    ys = [np.random.randint(1, 5, olen).astype(np.int32) for olen in olens]
    ilens = np.array([x.shape[0] for x in xs], dtype=np.int32)

    if mode == "chainer":
        if is_cuda:
            xp = importlib.import_module('cupy')
            xs = [chainer.Variable(xp.array(x)) for x in xs]
            ys = [chainer.Variable(xp.array(y)) for y in ys]
            ilens = xp.array(ilens)
        else:
            xs = [chainer.Variable(x) for x in xs]
            ys = [chainer.Variable(y) for y in ys]
        return xs, ilens, ys

    elif mode == "pytorch":
        ilens = torch.from_numpy(ilens).long()
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
        if is_cuda:
            xs_pad = xs_pad.cuda()
            ilens = ilens.cuda()
            ys_pad = ys_pad.cuda()

        return xs_pad, ilens, ys_pad
    else:
        raise ValueError("Invalid mode")


@pytest.mark.parametrize(
    "module, etype, atype", [
        ('espnet.nets.e2e_asr', 'vggblstmp', 'location'),
        ('espnet.nets.e2e_asr', 'blstmp', 'noatt'),
        ('espnet.nets.e2e_asr', 'blstmp', 'dot'),
        ('espnet.nets.e2e_asr', 'blstmp', 'location'),
        ('espnet.nets.e2e_asr_th', 'vggblstmp', 'location'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'noatt'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'dot'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'add'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'location'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'coverage'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'coverage_location'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'location2d'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'location_recurrent'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'multi_head_dot'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'multi_head_add'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'multi_head_loc'),
        ('espnet.nets.e2e_asr_th', 'blstmp', 'multi_head_multi_res_loc')
    ]
)
def test_model_trainable_and_decodable(module, etype, atype):
    args = make_arg(etype=etype, atype=atype)
    if module[-3:] == "_th":
        batch = prepare_inputs("pytorch")
    else:
        batch = prepare_inputs("chainer")

    m = importlib.import_module(module)
    model = m.Loss(m.E2E(40, 5, args), 0.5)
    attn_loss = model(*batch)
    attn_loss.backward()  # trainable

    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(100, 40)
        model.predictor.recognize(in_data, args, args.char_list)  # decodable


def init_torch_weight_const(m, val):
    for p in m.parameters():
        if p.dim() > 1:
            p.data.fill_(val)


def init_chainer_weight_const(m, val):
    for p in m.params():
        if p.data.ndim > 1:
            p.data[:] = val


def test_chainer_ctc_type():
    ch = importlib.import_module('espnet.nets.e2e_asr')
    np.random.seed(0)
    batch = prepare_inputs("chainer")

    def _propagate(ctc_type):
        args = make_arg(ctc_type=ctc_type)
        np.random.seed(0)
        model = ch.E2E(40, 5, args)
        ch_ctc, _, _ = model(*batch)
        ch_ctc.backward()
        W_grad = model.ctc.ctc_lo.W.grad
        b_grad = model.ctc.ctc_lo.b.grad
        return ch_ctc.data, W_grad, b_grad

    ref_loss, ref_W_grad, ref_b_grad = _propagate("chainer")
    loss, W_grad, b_grad = _propagate("warpctc")
    np.testing.assert_allclose(ref_loss, loss, rtol=1e-5)
    np.testing.assert_allclose(ref_W_grad, W_grad)
    np.testing.assert_allclose(ref_b_grad, b_grad)


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_loss_and_ctc_grad(etype):
    ch = importlib.import_module('espnet.nets.e2e_asr')
    th = importlib.import_module('espnet.nets.e2e_asr_th')
    args = make_arg(etype=etype)
    ch_model = ch.E2E(40, 5, args)
    ch_model.cleargrads()
    th_model = th.E2E(40, 5, args)

    const = 1e-4
    init_torch_weight_const(th_model, const)
    init_chainer_weight_const(ch_model, const)

    ch_batch = prepare_inputs("chainer")
    th_batch = prepare_inputs("pytorch")

    ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    th_ctc, th_att, th_acc, th_cer, th_wer = th_model(*th_batch)

    # test masking
    ch_ench = ch_model.att.pre_compute_enc_h.data
    th_ench = th_model.att.pre_compute_enc_h.detach().numpy()
    np.testing.assert_equal(ch_ench == 0.0, th_ench == 0.0)

    # test loss with constant weights (1.0) and bias (0.0) except for foget-bias (1.0)
    np.testing.assert_allclose(ch_ctc.data, th_ctc.detach().numpy())
    np.testing.assert_allclose(ch_att.data, th_att.detach().numpy())

    # test ctc grads
    ch_ctc.backward()
    th_ctc.backward()
    np.testing.assert_allclose(ch_model.ctc.ctc_lo.W.grad,
                               th_model.ctc.ctc_lo.weight.grad.data.numpy(), 1e-7, 1e-8)
    np.testing.assert_allclose(ch_model.ctc.ctc_lo.b.grad,
                               th_model.ctc.ctc_lo.bias.grad.data.numpy(), 1e-5, 1e-6)

    # test cross-entropy grads
    ch_model.cleargrads()
    th_model.zero_grad()

    ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    th_ctc, th_att, th_acc, th_cer, th_wer = th_model(*th_batch)
    ch_att.backward()
    th_att.backward()
    np.testing.assert_allclose(ch_model.dec.output.W.grad,
                               th_model.dec.output.weight.grad.data.numpy(), 1e-7, 1e-8)
    np.testing.assert_allclose(ch_model.dec.output.b.grad,
                               th_model.dec.output.bias.grad.data.numpy(), 1e-5, 1e-6)


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_mtl_loss(etype):
    ch = importlib.import_module('espnet.nets.e2e_asr')
    th = importlib.import_module('espnet.nets.e2e_asr_th')
    args = make_arg(etype=etype)
    ch_model = ch.E2E(40, 5, args)
    th_model = th.E2E(40, 5, args)

    const = 1e-4
    init_torch_weight_const(th_model, const)
    init_chainer_weight_const(ch_model, const)

    ch_batch = prepare_inputs("chainer")
    th_batch = prepare_inputs("pytorch")

    ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    th_ctc, th_att, th_acc, th_cer, th_wer = th_model(*th_batch)

    # test masking
    ch_ench = ch_model.att.pre_compute_enc_h.data
    th_ench = th_model.att.pre_compute_enc_h.detach().numpy()
    np.testing.assert_equal(ch_ench == 0.0, th_ench == 0.0)

    # test loss with constant weights (1.0) and bias (0.0) except for foget-bias (1.0)
    np.testing.assert_allclose(ch_ctc.data, th_ctc.detach().numpy())
    np.testing.assert_allclose(ch_att.data, th_att.detach().numpy())

    # test grads in mtl mode
    ch_loss = ch_ctc * 0.5 + ch_att * 0.5
    th_loss = th_ctc * 0.5 + th_att * 0.5
    ch_model.cleargrads()
    th_model.zero_grad()
    ch_loss.backward()
    th_loss.backward()
    np.testing.assert_allclose(ch_model.ctc.ctc_lo.W.grad,
                               th_model.ctc.ctc_lo.weight.grad.data.numpy(), 1e-7, 1e-8)
    np.testing.assert_allclose(ch_model.ctc.ctc_lo.b.grad,
                               th_model.ctc.ctc_lo.bias.grad.data.numpy(), 1e-5, 1e-6)
    np.testing.assert_allclose(ch_model.dec.output.W.grad,
                               th_model.dec.output.weight.grad.data.numpy(), 1e-7, 1e-8)
    np.testing.assert_allclose(ch_model.dec.output.b.grad,
                               th_model.dec.output.bias.grad.data.numpy(), 1e-5, 1e-6)


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_zero_length_target(etype):
    ch = importlib.import_module('espnet.nets.e2e_asr')
    th = importlib.import_module('espnet.nets.e2e_asr_th')
    args = make_arg(etype=etype)
    ch_model = ch.E2E(40, 5, args)
    ch_model.cleargrads()
    th_model = th.E2E(40, 5, args)

    ch_batch = prepare_inputs("chainer", olens=[4, 0])
    th_batch = prepare_inputs("pytorch", olens=[4, 0])

    ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    th_ctc, th_att, th_acc, th_cer, th_wer = th_model(*th_batch)

    # NOTE: We ignore all zero length case because chainer also fails. Have a nice data-prep!
    # out_data = ""
    # data = [
    #     ("aaa", dict(feat=np.random.randn(200, 40).astype(np.float32), tokenid="")),
    #     ("bbb", dict(feat=np.random.randn(100, 40).astype(np.float32), tokenid="")),
    #     ("cc", dict(feat=np.random.randn(100, 40).astype(np.float32), tokenid=""))
    # ]
    # ch_ctc, ch_att, ch_acc = ch_model(data)
    # th_ctc, th_att, th_acc = th_model(data)


@pytest.mark.parametrize(
    "module, atype", [
        ('espnet.nets.e2e_asr', 'noatt'),
        ('espnet.nets.e2e_asr', 'dot'),
        ('espnet.nets.e2e_asr', 'location'),
        ('espnet.nets.e2e_asr_th', 'noatt'),
        ('espnet.nets.e2e_asr_th', 'dot'),
        ('espnet.nets.e2e_asr_th', 'add'),
        ('espnet.nets.e2e_asr_th', 'location'),
        ('espnet.nets.e2e_asr_th', 'coverage'),
        ('espnet.nets.e2e_asr_th', 'coverage_location'),
        ('espnet.nets.e2e_asr_th', 'location2d'),
        ('espnet.nets.e2e_asr_th', 'location_recurrent'),
        ('espnet.nets.e2e_asr_th', 'multi_head_dot'),
        ('espnet.nets.e2e_asr_th', 'multi_head_add'),
        ('espnet.nets.e2e_asr_th', 'multi_head_loc'),
        ('espnet.nets.e2e_asr_th', 'multi_head_multi_res_loc')
    ]
)
def test_calculate_all_attentions(module, atype):
    m = importlib.import_module(module)
    args = make_arg(atype=atype)
    if module[-3:] == "_th":
        batch = prepare_inputs("pytorch")
    else:
        batch = prepare_inputs("chainer")
    model = m.E2E(40, 5, args)
    with chainer.no_backprop_mode():
        att_ws = model.calculate_all_attentions(*batch)
        print(att_ws.shape)


def test_chainer_save_and_load():
    m = importlib.import_module('espnet.nets.e2e_asr')
    utils = importlib.import_module('espnet.asr.asr_utils')
    args = make_arg()
    model = m.Loss(m.E2E(40, 5, args), 0.5)
    # initialize randomly
    for p in model.params():
        p.data = np.random.randn(*p.data.shape)
    tmppath = tempfile.mktemp()
    chainer.serializers.save_npz(tmppath, model)
    p_saved = [p.data for p in model.params()]
    # set constant value
    for p in model.params():
        p.data = np.zeros_like(p.data)
    utils.chainer_load(tmppath, model)
    for p1, p2 in zip(p_saved, model.params()):
        np.testing.assert_array_equal(p1, p2.data)
    if os.path.exists(tmppath):
        os.remove(tmppath)


def test_torch_save_and_load():
    m = importlib.import_module('espnet.nets.e2e_asr_th')
    utils = importlib.import_module('espnet.asr.asr_utils')
    args = make_arg()
    model = m.Loss(m.E2E(40, 5, args), 0.5)
    # initialize randomly
    for p in model.parameters():
        p.data.uniform_()
    if not os.path.exists(".pytest_cache"):
        os.makedirs(".pytest_cache")
    tmppath = tempfile.mktemp()
    utils.torch_save(tmppath, model)
    p_saved = [p.data.numpy() for p in model.parameters()]
    # set constant value
    for p in model.parameters():
        p.data.zero_()
    utils.torch_load(tmppath, model)
    for p1, p2 in zip(p_saved, model.parameters()):
        np.testing.assert_array_equal(p1, p2.data.numpy())
    if os.path.exists(tmppath):
        os.remove(tmppath)


@pytest.mark.skipif(not torch.cuda.is_available() and not chainer.cuda.available, reason="gpu required")
@pytest.mark.parametrize("module", ["espnet.nets.e2e_asr", "espnet.nets.e2e_asr_th"])
def test_gpu_trainable(module):
    m = importlib.import_module(module)
    args = make_arg()
    model = m.Loss(m.E2E(40, 5, args), 0.5)
    if module[-3:] == "_th":
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
    else:
        batch = prepare_inputs("chainer", is_cuda=True)
        model.to_gpu()
    loss = model(*batch)
    loss.backward()  # trainable


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
def test_torch_multi_gpu_trainable():
    m = importlib.import_module('espnet.nets.e2e_asr_th')
    ngpu = 2
    device_ids = list(range(ngpu))
    args = make_arg()
    model = m.Loss(m.E2E(40, 5, args), 0.5)
    model = torch.nn.DataParallel(model, device_ids)
    batch = prepare_inputs("pytorch", is_cuda=True)
    model.cuda()
    loss = 1. / ngpu * model(*batch)
    loss.backward(loss.new_ones(ngpu))  # trainable
