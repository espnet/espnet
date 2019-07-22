# coding: utf-8

# Copyright 2019 Hirofumi Inaguma
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
from espnet.utils.training.batchfy import make_batchset
from test.utils_test import make_dummy_json_mt


def make_arg(**kwargs):
    defaults = dict(
        elayers=2,
        subsample="2_2",
        etype="blstm",
        eunits=100,
        eprojs=100,
        dtype="lstm",
        dlayers=1,
        dunits=300,
        atype="add",
        aheads=4,
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
        ctc_weight=0.0,  # dummy
        ctc_window_margin=0,  # dummy
        verbose=2,
        char_list=[u"あ", u"い", u"う", u"え", u"お"],
        outdir=None,
        sym_space="<space>",
        sortagrad=0,
        context_residual=False,
        use_frontend=False,
        replace_sos=False,
        tgt_lang=False
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare_inputs(mode, ilens=[150, 100], olens=[4, 3], is_cuda=False):
    np.random.seed(1)
    assert len(ilens) == len(olens)
    xs = [np.random.randint(0, 5, ilen).astype(np.int32) for ilen in ilens]
    ys = [np.random.randint(0, 5, olen).astype(np.int32) for olen in olens]
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
        xs_pad = pad_list([torch.from_numpy(x).long() for x in xs], 6)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
        if is_cuda:
            xs_pad = xs_pad.cuda()
            ilens = ilens.cuda()
            ys_pad = ys_pad.cuda()

        return xs_pad, ilens, ys_pad
    else:
        raise ValueError("Invalid mode")


def convert_batch(batch, backend="pytorch", is_cuda=False, idim=5, odim=5):
    ilens = np.array([x[1]['output'][1]['shape'][0] for x in batch])
    olens = np.array([x[1]['output'][0]['shape'][0] for x in batch])
    xs = [np.random.randint(0, idim, ilen).astype(np.int32) for ilen in ilens]
    ys = [np.random.randint(0, odim, olen).astype(np.int32) for olen in olens]
    is_pytorch = backend == "pytorch"
    if is_pytorch:
        xs = pad_list([torch.from_numpy(x).long() for x in xs], idim)
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
        ('espnet.nets.pytorch_backend.e2e_mt', 'blstm', 'noatt', 'lstm'),  # Test Pytorch Attentions
        ('espnet.nets.pytorch_backend.e2e_mt', 'blstm', 'add', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'blstm', 'dot', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'blstm', 'coverage', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'blstm', 'multi_head_dot', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'blstm', 'multi_head_add', 'lstm'),

        ('espnet.nets.pytorch_backend.e2e_mt', 'grup', 'add', 'lstm'),  # Test Pytorch Encoders
        ('espnet.nets.pytorch_backend.e2e_mt', 'lstmp', 'add', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'bgrup', 'add', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'blstmp', 'add', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'bgru', 'add', 'lstm'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'blstm', 'add', 'lstm'),
    ]
)
def test_model_trainable_and_decodable(module, etype, atype, dtype):
    args = make_arg(etype=etype, atype=atype, dtype=dtype)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch")
    else:
        batch = prepare_inputs("chainer")

    m = importlib.import_module(module)
    model = m.E2E(6, 5, args)
    attn_loss = model(*batch)
    attn_loss.backward()  # trainable

    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randint(0, 5, (1, 100))
        model.translate(in_data, args, args.char_list)  # decodable
        if "pytorch" in module:
            batch_in_data = np.random.randint(0, 5, (2, 100))
            model.translate_batch(batch_in_data, args, args.char_list)  # batch decodable


@pytest.mark.parametrize(
    "module", ["pytorch"]
)
def test_sortagrad_trainable(module):
    args = make_arg(sortagrad=1)
    dummy_json = make_dummy_json_mt(8, [1, 100], [1, 100], idim=6, odim=5)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_mt as m
    else:
        import espnet.nets.chainer_backend.e2e_mt as m
    batchset = make_batchset(dummy_json, 2, 2 ** 10, 2 ** 10, shortest_first=True, mt=True)
    model = m.E2E(6, 5, args)
    for batch in batchset:
        attn_loss = model(*convert_batch(batch, module, idim=6, odim=5))
        attn_loss.backward()
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randint(0, 5, (1, 100))
        model.translate(in_data, args, args.char_list)


@pytest.mark.parametrize(
    "module", ["pytorch"]
)
def test_sortagrad_trainable_with_batch_bins(module):
    args = make_arg(sortagrad=1)
    idim = 6
    odim = 5
    dummy_json = make_dummy_json_mt(8, [100, 200], [100, 200], idim=idim, odim=odim)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_mt as m
    else:
        import espnet.nets.chainer_backend.e2e_mt as m
    batch_elems = 20000
    batchset = make_batchset(dummy_json, batch_bins=batch_elems, shortest_first=True, mt=True)
    for batch in batchset:
        n = 0
        for uttid, info in batch:
            ilen = int(info['output'][1]['shape'][0])
            olen = int(info['output'][0]['shape'][0])
            n += ilen * idim + olen * odim
        assert olen < batch_elems

    model = m.E2E(6, 5, args)
    for batch in batchset:
        attn_loss = model(*convert_batch(batch, module, idim=6, odim=5))
        attn_loss.backward()
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randint(0, 5, (1, 100))
        model.translate(in_data, args, args.char_list)


@pytest.mark.parametrize(
    "module", ["pytorch"]
)
def test_sortagrad_trainable_with_batch_frames(module):
    args = make_arg(sortagrad=1)
    idim = 6
    odim = 5
    dummy_json = make_dummy_json_mt(8, [100, 200], [100, 200], idim=idim, odim=odim)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_mt as m
    else:
        import espnet.nets.chainer_backend.e2e_mt as m
    batch_frames_in = 200
    batch_frames_out = 200
    batchset = make_batchset(dummy_json,
                             batch_frames_in=batch_frames_in,
                             batch_frames_out=batch_frames_out,
                             shortest_first=True,
                             mt=True)
    for batch in batchset:
        i = 0
        o = 0
        for uttid, info in batch:
            i += int(info['output'][1]['shape'][0])
            o += int(info['output'][0]['shape'][0])
        assert i <= batch_frames_in
        assert o <= batch_frames_out

    model = m.E2E(6, 5, args)
    for batch in batchset:
        attn_loss = model(*convert_batch(batch, module, idim=6, odim=5))
        attn_loss.backward()
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randint(0, 5, (1, 100))
        model.translate(in_data, args, args.char_list)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        if p.dim() > 1:
            p.data.fill_(val)


@pytest.mark.parametrize("etype", ["blstm"])
def test_loss(etype):
    # ch = importlib.import_module('espnet.nets.chainer_backend.e2e_mt')
    th = importlib.import_module('espnet.nets.pytorch_backend.e2e_mt')
    args = make_arg(etype=etype)
    # ch_model = ch.E2E(6, 5, args)
    # ch_model.cleargrads()
    th_model = th.E2E(6, 5, args)

    const = 1e-4
    init_torch_weight_const(th_model, const)
    # init_chainer_weight_const(ch_model, const)

    # ch_batch = prepare_inputs("chainer")
    th_batch = prepare_inputs("pytorch")

    # _, ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    th_model(*th_batch)
    th_att = th_model.loss

    # test masking
    # ch_ench = ch_model.att.pre_compute_enc_h.data
    # th_ench = th_model.att[0].pre_compute_enc_h.detach().numpy()
    # np.testing.assert_equal(ch_ench == 0.0, th_ench == 0.0)

    # test loss with constant weights (1.0) and bias (0.0) except for foget-bias (1.0)
    # np.testing.assert_allclose(ch_att.data, th_att.detach().numpy())

    # test cross-entropy grads
    # ch_model.cleargrads()
    th_model.zero_grad()

    # _, ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    th_model(*th_batch)
    th_att = th_model.loss
    # ch_att.backward()
    th_att.backward()
    # np.testing.assert_allclose(ch_model.dec.output.W.grad,
    #                            th_model.dec.output.weight.grad.data.numpy(), 1e-7, 1e-8)
    # np.testing.assert_allclose(ch_model.dec.output.b.grad,
    #                            th_model.dec.output.bias.grad.data.numpy(), 1e-5, 1e-6)


@pytest.mark.parametrize("etype", ["blstm"])
def test_zero_length_target(etype):
    # ch = importlib.import_module('espnet.nets.chainer_backend.e2e_mt')
    th = importlib.import_module('espnet.nets.pytorch_backend.e2e_mt')
    args = make_arg(etype=etype)
    # ch_model = ch.E2E(6, 5, args)
    # ch_model.cleargrads()
    th_model = th.E2E(6, 5, args)

    # ch_batch = prepare_inputs("chainer", olens=[4, 0])
    th_batch = prepare_inputs("pytorch", olens=[4, 0])

    # ch_model(*ch_batch)
    th_model(*th_batch)

    # NOTE: We ignore all zero length case because chainer also fails. Have a nice data-prep!
    # out_data = ""
    # data = [
    #     ("aaa", dict(feat=np.random.randint(0, 5, (1, 200)).astype(np.float32), tokenid="")),
    #     ("bbb", dict(feat=np.random.randint(0, 5, (1, 100)).astype(np.float32), tokenid="")),
    #     ("cc", dict(feat=np.random.randint(0, 5, (1, 100)).astype(np.float32), tokenid=""))
    # ]
    # ch_ctc, ch_att, ch_acc = ch_model(data)
    # th_ctc, th_att, th_acc = th_model(data)


@pytest.mark.parametrize(
    "module, atype", [
        ('espnet.nets.pytorch_backend.e2e_mt', 'noatt'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'dot'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'add'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'coverage'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'multi_head_dot'),
        ('espnet.nets.pytorch_backend.e2e_mt', 'multi_head_add'),
    ]
)
def test_calculate_all_attentions(module, atype):
    m = importlib.import_module(module)
    args = make_arg(atype=atype)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch")
    else:
        batch = prepare_inputs("chainer")
    model = m.E2E(6, 5, args)
    with chainer.no_backprop_mode():
        if "pytorch" in module:
            att_ws = model.calculate_all_attentions(*batch)[0]
        else:
            att_ws = model.calculate_all_attentions(*batch)
        print(att_ws.shape)


def test_torch_save_and_load():
    m = importlib.import_module('espnet.nets.pytorch_backend.e2e_mt')
    utils = importlib.import_module('espnet.asr.asr_utils')
    args = make_arg()
    model = m.E2E(6, 5, args)
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
@pytest.mark.parametrize("module", ["espnet.nets.pytorch_backend.e2e_mt"])
def test_gpu_trainable(module):
    m = importlib.import_module(module)
    args = make_arg()
    model = m.E2E(6, 5, args)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
    else:
        batch = prepare_inputs("chainer", is_cuda=True)
        model.to_gpu()
    loss = model(*batch)
    loss.backward()  # trainable


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize("module", ["espnet.nets.pytorch_backend.e2e_mt"])
def test_multi_gpu_trainable(module):
    m = importlib.import_module(module)
    ngpu = 2
    device_ids = list(range(ngpu))
    args = make_arg()
    model = m.E2E(6, 5, args)
    if "pytorch" in module:
        model = torch.nn.DataParallel(model, device_ids)
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
        loss = 1. / ngpu * model(*batch)
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
                loss = 1. / ngpu * _model(*batch)
                losses.append(loss)

        for loss in losses:
            loss.backward()  # trainable


@pytest.mark.parametrize(
    "module", ["pytorch"]
)
def test_context_residual(module):
    args = make_arg(context_residual=True)
    dummy_json = make_dummy_json_mt(8, [1, 100], [1, 100], idim=6, odim=5)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_mt as m
    else:
        raise NotImplementedError
    batchset = make_batchset(dummy_json, 2, 2 ** 10, 2 ** 10, shortest_first=True, mt=True)
    model = m.E2E(6, 5, args)
    for batch in batchset:
        attn_loss = model(*convert_batch(batch, module, idim=6, odim=5))
        attn_loss.backward()
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randint(0, 5, (1, 100))
        model.translate(in_data, args, args.char_list)
