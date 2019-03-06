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

from espnet.nets.pytorch_backend.nets_utils import pad_list
from test.utils_test import make_dummy_json


def make_arg(**kwargs):
    defaults = dict(
        elayers=4,
        subsample="1_2_2_1_1",
        etype="vggblstm",
        eunits=100,
        eprojs=100,
        dtype="lstm",
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
        dropout_rate_decoder=0.0,
        nbest=5,
        beam_size=3,
        penalty=0.5,
        maxlenratio=1.0,
        minlenratio=0.0,
        ctc_weight=0.2,
        verbose=2,
        char_list=[u"あ", u"い", u"う", u"え", u"お"],
        outdir=None,
        ctc_type="warpctc",
        sym_space="<space>",
        sym_blank="<blank>",
        sortagrad=0
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


def convert_batch(batch, backend="pytorch", is_cuda=False, idim=40, odim=5):
    ilens = np.array([x[1]['input'][0]['shape'][0] for x in batch])
    olens = np.array([x[1]['output'][0]['shape'][0] for x in batch])
    xs = [np.random.randn(ilen, idim).astype(np.float32) for ilen in ilens]
    ys = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens]
    is_pytorch = backend == "pytorch"
    if is_pytorch:
        xs = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ilens = torch.from_numpy(ilens).long()
        ys = pad_list([torch.from_numpy(y).long() for y in ys], -1)

        if is_cuda:
            xs = xs.cuda()
            ilens = ilens.cuda()
            ys = ys.cuda()
    else:
        if is_cuda:
            xp = importlib.import_module('cupy')
            xs = [chainer.Variable(xp.array(x)) for x in xs]
            ys = [chainer.Variable(xp.array(y)) for y in ys]
            ilens = xp.array(ilens)
        else:
            xs = [chainer.Variable(x) for x in xs]
            ys = [chainer.Variable(y) for y in ys]

    return xs, ilens, ys


@pytest.mark.parametrize(
    "module, etype, atype, dtype", [
        ('espnet.nets.chainer_backend.e2e_asr', 'vggblstmp', 'location', 'gru'),
        ('espnet.nets.chainer_backend.e2e_asr', 'blstmp', 'noatt', 'gru'),
        ('espnet.nets.chainer_backend.e2e_asr', 'blstmp', 'dot', 'gru'),
        ('espnet.nets.chainer_backend.e2e_asr', 'blstmp', 'location', 'gru'),
        ('espnet.nets.chainer_backend.e2e_asr', 'bgrup', 'location', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'vggblstmp', 'location', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'noatt', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'dot', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'add', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'location', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'coverage', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'coverage_location', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'location2d', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'location_recurrent', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'multi_head_dot', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'multi_head_add', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'multi_head_loc', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstmp', 'multi_head_multi_res_loc', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'blstm', 'location', 'gru'),
        ('espnet.nets.chainer_backend.e2e_asr', 'vggbgrup', 'location', 'lstm'),
        ('espnet.nets.chainer_backend.e2e_asr', 'bgrup', 'noatt', 'lstm'),
        ('espnet.nets.chainer_backend.e2e_asr', 'bgrup', 'dot', 'lstm'),
        ('espnet.nets.chainer_backend.e2e_asr', 'bgrup', 'location', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'vggbgrup', 'location', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'noatt', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'dot', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'add', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'location', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'coverage', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'coverage_location', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'location2d', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'location_recurrent', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'multi_head_dot', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'multi_head_add', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'multi_head_loc', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'multi_head_multi_res_loc', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgrup', 'multi_head_multi_res_loc', 'gru'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'bgru', 'location', 'lstm'),
    ]
)
def test_model_trainable_and_decodable(module, etype, atype, dtype):
    args = make_arg(etype=etype, atype=atype, dtype=dtype)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch")
    else:
        batch = prepare_inputs("chainer")

    m = importlib.import_module(module)
    model = m.E2E(40, 5, args)
    attn_loss = model(*batch)[0]
    attn_loss.backward()  # trainable

    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(100, 40)
        model.recognize(in_data, args, args.char_list)  # decodable


def test_streaming_e2e():
    m = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr')
    args = make_arg()
    model = m.E2E(40, 5, args)
    asr = m.StreamingE2E(model, args, args.char_list)

    in_data = np.random.randn(100, 40)
    for i in range(10):
        asr.accept_input(in_data)

    asr.advance_decoder()

    recogs = asr.retrieve_recognition()
    pass


@pytest.mark.parametrize(
    "module", ["pytorch", "chainer"]
)
def test_sortagrad_trainable(module):
    args = make_arg(sortagrad=1)
    dummy_json = make_dummy_json(8, [1, 700], [1, 700], idim=20, odim=5)
    from espnet.asr.asr_utils import make_batchset
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_asr as m
    else:
        import espnet.nets.chainer_backend.e2e_asr as m
    batchset = make_batchset(dummy_json, 2, 2 ** 10, 2 ** 10, shortest_first=True)
    model = m.E2E(20, 5, args)
    for batch in batchset:
        attn_loss = model(*convert_batch(batch, module, idim=20, odim=5))[0]
        attn_loss.backward()
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(100, 20)
        model.recognize(in_data, args, args.char_list)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        if p.dim() > 1:
            p.data.fill_(val)


def init_chainer_weight_const(m, val):
    for p in m.params():
        if p.data.ndim > 1:
            p.data[:] = val


def test_chainer_ctc_type():
    ch = importlib.import_module('espnet.nets.chainer_backend.e2e_asr')
    np.random.seed(0)
    batch = prepare_inputs("chainer")

    def _propagate(ctc_type):
        args = make_arg(ctc_type=ctc_type)
        np.random.seed(0)
        model = ch.E2E(40, 5, args)
        _, ch_ctc, _, _ = model(*batch)
        ch_ctc.backward()
        W_grad = model.ctc.ctc_lo.W.grad
        b_grad = model.ctc.ctc_lo.b.grad
        return ch_ctc.data, W_grad, b_grad

    ref_loss, ref_W_grad, ref_b_grad = _propagate("builtin")
    loss, W_grad, b_grad = _propagate("warpctc")
    np.testing.assert_allclose(ref_loss, loss, rtol=1e-5)
    np.testing.assert_allclose(ref_W_grad, W_grad)
    np.testing.assert_allclose(ref_b_grad, b_grad)


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_loss_and_ctc_grad(etype):
    ch = importlib.import_module('espnet.nets.chainer_backend.e2e_asr')
    th = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr')
    args = make_arg(etype=etype)
    ch_model = ch.E2E(40, 5, args)
    ch_model.cleargrads()
    th_model = th.E2E(40, 5, args)

    const = 1e-4
    init_torch_weight_const(th_model, const)
    init_chainer_weight_const(ch_model, const)

    ch_batch = prepare_inputs("chainer")
    th_batch = prepare_inputs("pytorch")

    _, ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    _, th_ctc, th_att, th_acc, th_cer, th_wer = th_model(*th_batch)

    # test masking
    ch_ench = ch_model.att.pre_compute_enc_h.data
    th_ench = th_model.att[0].pre_compute_enc_h.detach().numpy()
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

    _, ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    _, th_ctc, th_att, th_acc, th_cer, th_wer = th_model(*th_batch)
    ch_att.backward()
    th_att.backward()
    np.testing.assert_allclose(ch_model.dec.output.W.grad,
                               th_model.dec.output.weight.grad.data.numpy(), 1e-7, 1e-8)
    np.testing.assert_allclose(ch_model.dec.output.b.grad,
                               th_model.dec.output.bias.grad.data.numpy(), 1e-5, 1e-6)


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_mtl_loss(etype):
    ch = importlib.import_module('espnet.nets.chainer_backend.e2e_asr')
    th = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr')
    args = make_arg(etype=etype)
    ch_model = ch.E2E(40, 5, args)
    th_model = th.E2E(40, 5, args)

    const = 1e-4
    init_torch_weight_const(th_model, const)
    init_chainer_weight_const(ch_model, const)

    ch_batch = prepare_inputs("chainer")
    th_batch = prepare_inputs("pytorch")

    _, ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    _, th_ctc, th_att, th_acc, th_cer, th_wer = th_model(*th_batch)

    # test masking
    ch_ench = ch_model.att.pre_compute_enc_h.data
    th_ench = th_model.att[0].pre_compute_enc_h.detach().numpy()
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
    ch = importlib.import_module('espnet.nets.chainer_backend.e2e_asr')
    th = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr')
    args = make_arg(etype=etype)
    ch_model = ch.E2E(40, 5, args)
    ch_model.cleargrads()
    th_model = th.E2E(40, 5, args)

    ch_batch = prepare_inputs("chainer", olens=[4, 0])
    th_batch = prepare_inputs("pytorch", olens=[4, 0])

    _, ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    _, th_ctc, th_att, th_acc, th_cer, th_wer = th_model(*th_batch)

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
        ('espnet.nets.chainer_backend.e2e_asr', 'noatt'),
        ('espnet.nets.chainer_backend.e2e_asr', 'dot'),
        ('espnet.nets.chainer_backend.e2e_asr', 'location'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'noatt'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'dot'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'add'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'location'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'coverage'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'coverage_location'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'location2d'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'location_recurrent'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'multi_head_dot'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'multi_head_add'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'multi_head_loc'),
        ('espnet.nets.pytorch_backend.e2e_asr', 'multi_head_multi_res_loc')
    ]
)
def test_calculate_all_attentions(module, atype):
    m = importlib.import_module(module)
    args = make_arg(atype=atype)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch")
    else:
        batch = prepare_inputs("chainer")
    model = m.E2E(40, 5, args)
    with chainer.no_backprop_mode():
        if "pytorch" in module:
            att_ws = model.calculate_all_attentions(*batch)[0]
        else:
            att_ws = model.calculate_all_attentions(*batch)
        print(att_ws.shape)


def test_chainer_save_and_load():
    m = importlib.import_module('espnet.nets.chainer_backend.e2e_asr')
    utils = importlib.import_module('espnet.asr.asr_utils')
    args = make_arg()
    model = m.E2E(40, 5, args)
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
    m = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr')
    utils = importlib.import_module('espnet.asr.asr_utils')
    args = make_arg()
    model = m.E2E(40, 5, args)
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
@pytest.mark.parametrize("module", ["espnet.nets.chainer_backend.e2e_asr", "espnet.nets.pytorch_backend.e2e_asr"])
def test_gpu_trainable(module):
    m = importlib.import_module(module)
    args = make_arg()
    model = m.E2E(40, 5, args)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
    else:
        batch = prepare_inputs("chainer", is_cuda=True)
        model.to_gpu()
    loss = model(*batch)[0]
    loss.backward()  # trainable


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize("module", ["espnet.nets.chainer_backend.e2e_asr", "espnet.nets.pytorch_backend.e2e_asr"])
def test_multi_gpu_trainable(module):
    m = importlib.import_module(module)
    ngpu = 2
    device_ids = list(range(ngpu))
    args = make_arg()
    model = m.E2E(40, 5, args)
    if "pytorch" in module:
        model = torch.nn.DataParallel(model, device_ids)
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
        loss = 1. / ngpu * model(*batch)[0]
        loss.backward(loss.new_ones(ngpu))  # trainable
    else:
        import copy
        import cupy
        losses = []
        for device in device_ids:
            with cupy.cuda.Device(device):
                batch = prepare_inputs("chainer", is_cuda=True)
                _model = copy.deepcopy(model)  # Transcribed from training.updaters.ParallelUpdater
                _model.to_gpu()
                loss = 1. / ngpu * _model(*batch)[0]
                losses.append(loss)

        for loss in losses:
            loss.backward()  # trainable
