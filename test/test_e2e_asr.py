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

from espnet.asr import asr_utils
import espnet.nets.chainer_backend.e2e_asr as ch_asr
import espnet.nets.pytorch_backend.e2e_asr as th_asr
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
from espnet.utils.training.batchfy import make_batchset
from test.utils_test import make_dummy_json


def make_arg(**kwargs):
    defaults = dict(
        elayers=1,
        subsample="1_2_2_1_1",
        etype="vggblstm",
        eunits=2,
        eprojs=2,
        dtype="lstm",
        dlayers=1,
        dunits=2,
        atype="location",
        aheads=1,
        awin=3,
        aconv_chans=1,
        aconv_filts=1,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=2,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        nbest=3,
        beam_size=2,
        penalty=0.5,
        maxlenratio=1.0,
        minlenratio=0.0,
        ctc_weight=0.2,
        ctc_window_margin=0,
        lm_weight=0.0,
        rnnlm=None,
        streaming_min_blank_dur=10,
        streaming_onset_margin=2,
        streaming_offset_margin=2,
        verbose=2,
        char_list=["あ", "い", "う", "え", "お"],
        outdir=None,
        ctc_type="warpctc",
        report_cer=False,
        report_wer=False,
        sym_space="<space>",
        sym_blank="<blank>",
        sortagrad=0,
        grad_noise=False,
        context_residual=False,
        use_frontend=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare_inputs(mode, ilens=[14, 13], olens=[4, 3], is_cuda=False):
    np.random.seed(1)
    assert len(ilens) == len(olens)
    xs = [np.random.randn(ilen, 10).astype(np.float32) for ilen in ilens]
    ys = [np.random.randint(1, 5, olen).astype(np.int32) for olen in olens]
    ilens = np.array([x.shape[0] for x in xs], dtype=np.int32)

    if mode == "chainer":
        if is_cuda:
            xp = importlib.import_module("cupy")
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


def convert_batch(batch, backend="pytorch", is_cuda=False, idim=10, odim=5):
    ilens = np.array([x[1]["input"][0]["shape"][0] for x in batch])
    olens = np.array([x[1]["output"][0]["shape"][0] for x in batch])
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
            xp = importlib.import_module("cupy")
            xs = [chainer.Variable(xp.array(x)) for x in xs]
            ys = [chainer.Variable(xp.array(y)) for y in ys]
            ilens = xp.array(ilens)
        else:
            xs = [chainer.Variable(x) for x in xs]
            ys = [chainer.Variable(y) for y in ys]

    return xs, ilens, ys


@pytest.mark.parametrize(
    "module, model_dict",
    [
        ("espnet.nets.chainer_backend.e2e_asr", {}),
        ("espnet.nets.chainer_backend.e2e_asr", {"elayers": 2, "dlayers": 2}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "vggblstmp"}),
        (
            "espnet.nets.chainer_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "noatt"},
        ),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "vggblstmp", "atype": "dot"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "grup"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "lstmp"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "bgrup"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "blstmp"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "bgru"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "blstm"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "vgggru"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "vggbgrup"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "vgglstm"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "vgglstmp"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "vggbgru"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "vggbgrup"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"etype": "vggblstmp", "dtype": "gru"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"mtlalpha": 0.0}),
        ("espnet.nets.chainer_backend.e2e_asr", {"mtlalpha": 1.0}),
        ("espnet.nets.chainer_backend.e2e_asr", {"sampling_probability": 0.5}),
        ("espnet.nets.chainer_backend.e2e_asr", {"ctc_type": "builtin"}),
        ("espnet.nets.chainer_backend.e2e_asr", {"ctc_weight": 0.0}),
        ("espnet.nets.chainer_backend.e2e_asr", {"ctc_weight": 1.0}),
        ("espnet.nets.chainer_backend.e2e_asr", {"report_cer": True}),
        ("espnet.nets.chainer_backend.e2e_asr", {"report_wer": True}),
        (
            "espnet.nets.chainer_backend.e2e_asr",
            {"report_cer": True, "report_wer": True},
        ),
        (
            "espnet.nets.chainer_backend.e2e_asr",
            {"report_cer": True, "report_wer": True, "mtlalpha": 0.0},
        ),
        (
            "espnet.nets.chainer_backend.e2e_asr",
            {"report_cer": True, "report_wer": True, "mtlalpha": 1.0},
        ),
        ("espnet.nets.pytorch_backend.e2e_asr", {}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"elayers": 2, "dlayers": 2}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "grup"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "lstmp"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "bgrup"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "blstmp"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "bgru"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "blstm"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "vgggru"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "vgggrup"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "vgglstm"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "vgglstmp"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "vggbgru"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "vggbgrup"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "vggblstmp", "dtype": "gru"}),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "noatt"},
        ),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "vggblstmp", "atype": "add"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"etype": "vggblstmp", "atype": "dot"}),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "coverage"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "coverage_location"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "location2d"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "location_recurrent"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "multi_head_dot"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "multi_head_add"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "multi_head_loc"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"etype": "vggblstmp", "atype": "multi_head_multi_res_loc"},
        ),
        ("espnet.nets.pytorch_backend.e2e_asr", {"mtlalpha": 0.0}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"mtlalpha": 1.0}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"sampling_probability": 0.5}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"ctc_type": "builtin"}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"ctc_weight": 0.0}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"ctc_weight": 1.0}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"context_residual": True}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"grad_noise": True}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"report_cer": True}),
        ("espnet.nets.pytorch_backend.e2e_asr", {"report_wer": True}),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"report_cer": True, "report_wer": True},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"report_cer": True, "report_wer": True, "mtlalpha": 0.0},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr",
            {"report_cer": True, "report_wer": True, "mtlalpha": 1.0},
        ),
    ],
)
def test_model_trainable_and_decodable(module, model_dict):
    args = make_arg(**model_dict)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch")
        m = th_asr
    else:
        batch = prepare_inputs("chainer")
        m = ch_asr

    model = m.E2E(10, 5, args)
    loss = model(*batch)
    if isinstance(loss, tuple):
        # chainer return several values as tuple
        loss[0].backward()  # trainable
    else:
        loss.backward()  # trainable

    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(10, 10)
        model.recognize(in_data, args, args.char_list)  # decodable
        if "pytorch" in module:
            batch_in_data = [np.random.randn(10, 10), np.random.randn(5, 10)]
            model.recognize_batch(
                batch_in_data, args, args.char_list
            )  # batch decodable


def test_window_streaming_e2e_encoder_and_ctc_with_offline_attention():
    args = make_arg()
    model = th_asr.E2E(10, 5, args)
    asr = WindowStreamingE2E(model, args)
    in_data = np.random.randn(100, 10)
    for i in range(2):
        asr.accept_input(in_data)
    asr.decode_with_attention_offline()


def test_segment_streaming_e2e():
    args = make_arg()
    args.etype = "vgglstm"  # uni-directional
    args.batchsize = 0
    model = th_asr.E2E(10, 5, args)
    asr = SegmentStreamingE2E(model, args)

    in_data = np.random.randn(50, 10)
    r = np.prod(model.subsample)
    for i in range(0, 50, r):
        asr.accept_input(in_data[i : i + r])

    args.batchsize = 1
    for i in range(0, 50, r):
        asr.accept_input(in_data[i : i + r])


@pytest.mark.parametrize("module", ["pytorch"])
def test_gradient_noise_injection(module):
    args = make_arg(grad_noise=True)
    args_org = make_arg()
    dummy_json = make_dummy_json(2, [3, 4], [3, 4], idim=10, odim=5)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_asr as m
    else:
        import espnet.nets.chainer_backend.e2e_asr as m
    batchset = make_batchset(dummy_json, 2, 2**10, 2**10, shortest_first=True)
    model = m.E2E(10, 5, args)
    model_org = m.E2E(10, 5, args_org)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=10, odim=5))
        loss_org = model_org(*convert_batch(batch, module, idim=10, odim=5))
        loss.backward()
        grad = [param.grad for param in model.parameters()][10]
        loss_org.backward()
        grad_org = [param.grad for param in model_org.parameters()][10]
        assert grad[0] != grad_org[0]


@pytest.mark.parametrize("module", ["pytorch", "chainer"])
def test_sortagrad_trainable(module):
    args = make_arg(sortagrad=1)
    idim = 10
    odim = 5
    dummy_json = make_dummy_json(2, [3, 5], [3, 5], idim=idim, odim=odim)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_asr as m
    else:
        import espnet.nets.chainer_backend.e2e_asr as m
    batchset = make_batchset(dummy_json, 2, 2**10, 2**10, shortest_first=True)
    model = m.E2E(idim, odim, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=idim, odim=odim))
        if isinstance(loss, tuple):
            # chainer return several values as tuple
            loss[0].backward()  # trainable
        else:
            loss.backward()  # trainable
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(10, idim)
        model.recognize(in_data, args, args.char_list)


@pytest.mark.parametrize("module", ["pytorch", "chainer"])
def test_sortagrad_trainable_with_batch_bins(module):
    args = make_arg(sortagrad=1)
    idim = 10
    odim = 5
    dummy_json = make_dummy_json(2, [3, 5], [3, 5], idim=idim, odim=odim)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_asr as m
    else:
        import espnet.nets.chainer_backend.e2e_asr as m
    batch_elems = 2000
    batchset = make_batchset(dummy_json, batch_bins=batch_elems, shortest_first=True)
    for batch in batchset:
        n = 0
        for uttid, info in batch:
            ilen = int(info["input"][0]["shape"][0])
            olen = int(info["output"][0]["shape"][0])
            n += ilen * idim + olen * odim
        assert olen < batch_elems

    model = m.E2E(idim, odim, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=idim, odim=odim))
        if isinstance(loss, tuple):
            # chainer return several values as tuple
            loss[0].backward()  # trainable
        else:
            loss.backward()  # trainable
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(10, idim)
        model.recognize(in_data, args, args.char_list)


@pytest.mark.parametrize("module", ["pytorch", "chainer"])
def test_sortagrad_trainable_with_batch_frames(module):
    args = make_arg(sortagrad=1)
    idim = 10
    odim = 5
    dummy_json = make_dummy_json(2, [3, 5], [3, 5], idim=idim, odim=odim)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_asr as m
    else:
        import espnet.nets.chainer_backend.e2e_asr as m
    batch_frames_in = 50
    batch_frames_out = 50
    batchset = make_batchset(
        dummy_json,
        batch_frames_in=batch_frames_in,
        batch_frames_out=batch_frames_out,
        shortest_first=True,
    )
    for batch in batchset:
        i = 0
        o = 0
        for uttid, info in batch:
            i += int(info["input"][0]["shape"][0])
            o += int(info["output"][0]["shape"][0])
        assert i <= batch_frames_in
        assert o <= batch_frames_out

    model = m.E2E(idim, odim, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=idim, odim=odim))
        if isinstance(loss, tuple):
            # chainer return several values as tuple
            loss[0].backward()  # trainable
        else:
            loss.backward()  # trainable
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(10, idim)
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
    np.random.seed(0)
    batch = prepare_inputs("chainer")

    def _propagate(ctc_type):
        args = make_arg(ctc_type=ctc_type)
        np.random.seed(0)
        model = ch_asr.E2E(10, 5, args)
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


def test_loss_and_ctc_grad():
    args = make_arg(etype="vggblstmp")
    ch_model = ch_asr.E2E(10, 5, args)
    ch_model.cleargrads()
    th_model = th_asr.E2E(10, 5, args)

    const = 1e-4
    init_torch_weight_const(th_model, const)
    init_chainer_weight_const(ch_model, const)

    ch_batch = prepare_inputs("chainer")
    th_batch = prepare_inputs("pytorch")

    _, ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    th_model(*th_batch)
    th_ctc, th_att = th_model.loss_ctc, th_model.loss_att

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
    np.testing.assert_allclose(
        ch_model.ctc.ctc_lo.W.grad,
        th_model.ctc.ctc_lo.weight.grad.data.numpy(),
        1e-7,
        1e-8,
    )
    np.testing.assert_allclose(
        ch_model.ctc.ctc_lo.b.grad,
        th_model.ctc.ctc_lo.bias.grad.data.numpy(),
        1e-5,
        1e-6,
    )

    # test cross-entropy grads
    ch_model.cleargrads()
    th_model.zero_grad()

    _, ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    th_model(*th_batch)
    th_ctc, th_att = th_model.loss_ctc, th_model.loss_att
    ch_att.backward()
    th_att.backward()
    np.testing.assert_allclose(
        ch_model.dec.output.W.grad,
        th_model.dec.output.weight.grad.data.numpy(),
        1e-7,
        1e-8,
    )
    np.testing.assert_allclose(
        ch_model.dec.output.b.grad,
        th_model.dec.output.bias.grad.data.numpy(),
        1e-5,
        1e-6,
    )


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_mtl_loss(etype):
    args = make_arg(etype=etype)
    ch_model = ch_asr.E2E(10, 5, args)
    th_model = th_asr.E2E(10, 5, args)

    const = 1e-4
    init_torch_weight_const(th_model, const)
    init_chainer_weight_const(ch_model, const)

    ch_batch = prepare_inputs("chainer")
    th_batch = prepare_inputs("pytorch")

    _, ch_ctc, ch_att, ch_acc = ch_model(*ch_batch)
    th_model(*th_batch)
    th_ctc, th_att = th_model.loss_ctc, th_model.loss_att

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
    np.testing.assert_allclose(
        ch_model.ctc.ctc_lo.W.grad,
        th_model.ctc.ctc_lo.weight.grad.data.numpy(),
        1e-7,
        1e-8,
    )
    np.testing.assert_allclose(
        ch_model.ctc.ctc_lo.b.grad,
        th_model.ctc.ctc_lo.bias.grad.data.numpy(),
        1e-5,
        1e-6,
    )
    np.testing.assert_allclose(
        ch_model.dec.output.W.grad,
        th_model.dec.output.weight.grad.data.numpy(),
        1e-7,
        1e-8,
    )
    np.testing.assert_allclose(
        ch_model.dec.output.b.grad,
        th_model.dec.output.bias.grad.data.numpy(),
        1e-5,
        1e-6,
    )


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_zero_length_target(etype):
    args = make_arg(etype=etype)
    ch_model = ch_asr.E2E(10, 5, args)
    ch_model.cleargrads()
    th_model = th_asr.E2E(10, 5, args)

    ch_batch = prepare_inputs("chainer", olens=[4, 0])
    th_batch = prepare_inputs("pytorch", olens=[4, 0])

    ch_model(*ch_batch)
    th_model(*th_batch)

    # NOTE: We ignore all zero length case because chainer also fails.
    # Have a nice data-prep!
    # out_data = ""
    # data = [
    #     ("aaa", dict(feat=np.random.randn(200, 10).astype(np.float32), tokenid="")),
    #     ("bbb", dict(feat=np.random.randn(100, 10).astype(np.float32), tokenid="")),
    #     ("cc", dict(feat=np.random.randn(100, 10).astype(np.float32), tokenid=""))
    # ]
    # ch_ctc, ch_att, ch_acc = ch_model(data)
    # th_ctc, th_att, th_acc = th_model(data)


@pytest.mark.parametrize(
    "module, atype",
    [
        ("espnet.nets.chainer_backend.e2e_asr", "noatt"),
        ("espnet.nets.chainer_backend.e2e_asr", "dot"),
        ("espnet.nets.chainer_backend.e2e_asr", "location"),
        ("espnet.nets.pytorch_backend.e2e_asr", "noatt"),
        ("espnet.nets.pytorch_backend.e2e_asr", "dot"),
        ("espnet.nets.pytorch_backend.e2e_asr", "add"),
        ("espnet.nets.pytorch_backend.e2e_asr", "location"),
        ("espnet.nets.pytorch_backend.e2e_asr", "coverage"),
        ("espnet.nets.pytorch_backend.e2e_asr", "coverage_location"),
        ("espnet.nets.pytorch_backend.e2e_asr", "location2d"),
        ("espnet.nets.pytorch_backend.e2e_asr", "location_recurrent"),
        ("espnet.nets.pytorch_backend.e2e_asr", "multi_head_dot"),
        ("espnet.nets.pytorch_backend.e2e_asr", "multi_head_add"),
        ("espnet.nets.pytorch_backend.e2e_asr", "multi_head_loc"),
        ("espnet.nets.pytorch_backend.e2e_asr", "multi_head_multi_res_loc"),
    ],
)
def test_calculate_all_attentions(module, atype):
    m = importlib.import_module(module)
    args = make_arg(atype=atype)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch")
    else:
        batch = prepare_inputs("chainer")
    model = m.E2E(10, 5, args)
    with chainer.no_backprop_mode():
        if "pytorch" in module:
            att_ws = model.calculate_all_attentions(*batch)[0]
        else:
            att_ws = model.calculate_all_attentions(*batch)
        print(att_ws.shape)


@pytest.mark.parametrize("mtlalpha", [0.0, 0.5, 1.0])
def test_calculate_all_ctc_probs(mtlalpha):
    args = make_arg(mtlalpha=mtlalpha)
    batch = prepare_inputs("pytorch")
    model = th_asr.E2E(10, 5, args)
    ctc_probs = model.calculate_all_ctc_probs(*batch)
    if mtlalpha > 0:
        print(ctc_probs.shape)
    else:
        assert ctc_probs is None


def test_chainer_save_and_load():
    args = make_arg()
    model = ch_asr.E2E(10, 5, args)
    # initialize randomly
    for p in model.params():
        p.data = np.random.randn(*p.data.shape)
    tmppath = tempfile.mktemp()
    chainer.serializers.save_npz(tmppath, model)
    p_saved = [p.data for p in model.params()]
    # set constant value
    for p in model.params():
        p.data = np.zeros_like(p.data)
    asr_utils.chainer_load(tmppath, model)
    for p1, p2 in zip(p_saved, model.params()):
        np.testing.assert_array_equal(p1, p2.data)
    if os.path.exists(tmppath):
        os.remove(tmppath)


def test_torch_save_and_load():
    args = make_arg()
    model = th_asr.E2E(10, 5, args)
    # initialize randomly
    for p in model.parameters():
        p.data.uniform_()
    if not os.path.exists(".pytest_cache"):
        os.makedirs(".pytest_cache")
    tmppath = tempfile.mktemp()
    asr_utils.torch_save(tmppath, model)
    p_saved = [p.data.numpy() for p in model.parameters()]
    # set constant value
    for p in model.parameters():
        p.data.zero_()
    asr_utils.torch_load(tmppath, model)
    for p1, p2 in zip(p_saved, model.parameters()):
        np.testing.assert_array_equal(p1, p2.data.numpy())
    if os.path.exists(tmppath):
        os.remove(tmppath)


@pytest.mark.skipif(
    not torch.cuda.is_available() and not chainer.cuda.available, reason="gpu required"
)
@pytest.mark.parametrize(
    "module",
    ["espnet.nets.chainer_backend.e2e_asr", "espnet.nets.pytorch_backend.e2e_asr"],
)
def test_gpu_trainable(module):
    m = importlib.import_module(module)
    args = make_arg()
    model = m.E2E(10, 5, args)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
    else:
        batch = prepare_inputs("chainer", is_cuda=True)
        model.to_gpu()
    loss = model(*batch)
    if isinstance(loss, tuple):
        # chainer return several values as tuple
        loss[0].backward()  # trainable
    else:
        loss.backward()  # trainable


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize(
    "module",
    ["espnet.nets.chainer_backend.e2e_asr", "espnet.nets.pytorch_backend.e2e_asr"],
)
def test_multi_gpu_trainable(module):
    m = importlib.import_module(module)
    ngpu = 2
    device_ids = list(range(ngpu))
    args = make_arg()
    model = m.E2E(10, 5, args)
    if "pytorch" in module:
        model = torch.nn.DataParallel(model, device_ids)
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
        loss = 1.0 / ngpu * model(*batch)
        loss.backward(loss.new_ones(ngpu))  # trainable
    else:
        import copy
        import cupy

        losses = []
        for device in device_ids:
            with cupy.cuda.Device(device):
                batch = prepare_inputs("chainer", is_cuda=True)
                _model = copy.deepcopy(
                    model
                )  # Transcribed from training.updaters.ParallelUpdater
                _model.to_gpu()
                loss = 1.0 / ngpu * _model(*batch)[0]
                losses.append(loss)

        for loss in losses:
            loss.backward()  # trainable
