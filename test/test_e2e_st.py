# coding: utf-8

# Copyright 2019 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import importlib
import os
import tempfile
from test.utils_test import make_dummy_json_st

import chainer
import numpy as np
import pytest
import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.utils.training.batchfy import make_batchset


def make_arg(**kwargs):
    defaults = dict(
        elayers=1,
        subsample="1_2_2_1_1",
        etype="vggblstm",
        eunits=16,
        eprojs=8,
        dtype="lstm",
        dlayers=1,
        dunits=16,
        atype="add",
        aheads=2,
        awin=5,
        aconv_chans=4,
        aconv_filts=10,
        mtlalpha=0.0,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=16,
        dropout_rate=0.0,
        dropout_rate_decoder=0.0,
        nbest=5,
        beam_size=2,
        penalty=0.5,
        maxlenratio=1.0,
        minlenratio=0.0,
        ctc_weight=0.0,
        ctc_window_margin=0,  # dummy
        lm_weight=0.0,
        rnnlm=None,
        streaming_min_blank_dur=10,
        streaming_onset_margin=2,
        streaming_offset_margin=2,
        verbose=2,
        char_list=["あ", "い", "う", "え", "お"],
        outdir=None,
        ctc_type="warpctc",
        report_bleu=False,
        report_cer=False,
        report_wer=False,
        sym_space="<space>",
        sym_blank="<blank>",
        sortagrad=0,
        grad_noise=False,
        context_residual=False,
        multilingual=False,
        replace_sos=False,
        tgt_lang=False,
        asr_weight=0.0,
        mt_weight=0.0,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare_inputs(
    mode, ilens=[20, 15], olens_tgt=[4, 3], olens_src=[3, 2], is_cuda=False
):
    np.random.seed(1)
    assert len(ilens) == len(olens_tgt)
    xs = [np.random.randn(ilen, 40).astype(np.float32) for ilen in ilens]
    ys_tgt = [np.random.randint(1, 5, olen).astype(np.int32) for olen in olens_tgt]
    ys_src = [np.random.randint(1, 5, olen).astype(np.int32) for olen in olens_src]
    ilens = np.array([x.shape[0] for x in xs], dtype=np.int32)

    if mode == "chainer":
        raise NotImplementedError

    elif mode == "pytorch":
        ilens = torch.from_numpy(ilens).long()
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ys_pad_tgt = pad_list([torch.from_numpy(y).long() for y in ys_tgt], -1)
        ys_pad_src = pad_list([torch.from_numpy(y).long() for y in ys_src], -1)
        if is_cuda:
            xs_pad = xs_pad.cuda()
            ilens = ilens.cuda()
            ys_pad_tgt = ys_pad_tgt.cuda()
            ys_pad_src = ys_pad_src.cuda()

        return xs_pad, ilens, ys_pad_tgt, ys_pad_src
    else:
        raise ValueError("Invalid mode")


def convert_batch(batch, backend="pytorch", is_cuda=False, idim=40, odim=5):
    ilens = np.array([x[1]["input"][0]["shape"][0] for x in batch])
    olens_tgt = np.array([x[1]["output"][0]["shape"][0] for x in batch])
    olens_src = np.array([x[1]["output"][1]["shape"][0] for x in batch])
    xs = [np.random.randn(ilen, idim).astype(np.float32) for ilen in ilens]
    ys_tgt = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens_tgt]
    ys_src = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens_src]
    is_pytorch = backend == "pytorch"
    if is_pytorch:
        xs = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ilens = torch.from_numpy(ilens).long()
        ys_tgt = pad_list([torch.from_numpy(y).long() for y in ys_tgt], -1)
        ys_src = pad_list([torch.from_numpy(y).long() for y in ys_src], -1)

        if is_cuda:
            xs = xs.cuda()
            ilens = ilens.cuda()
            ys_tgt = ys_tgt.cuda()
            ys_src = ys_src.cuda()
    else:
        raise NotImplementedError

    return xs, ilens, ys_tgt, ys_src


@pytest.mark.parametrize(
    "module, model_dict",
    [
        ("espnet.nets.pytorch_backend.e2e_st", {}),
        ("espnet.nets.pytorch_backend.e2e_st", {"elayers": 2, "dlayers": 2}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "grup"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "lstmp"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "bgrup"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "blstmp"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "bgru"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "blstm"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "vgggru"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "vgggrup"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "vgglstm"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "vgglstmp"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "vggbgru"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "vggbgrup"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "vggblstmp", "dtype": "gru"}),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"etype": "vggblstmp", "atype": "noatt"},
        ),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "vggblstmp", "atype": "add"}),
        ("espnet.nets.pytorch_backend.e2e_st", {"etype": "vggblstmp", "atype": "dot"}),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"etype": "vggblstmp", "atype": "coverage"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"etype": "vggblstmp", "atype": "coverage_location"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"etype": "vggblstmp", "atype": "location2d"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"etype": "vggblstmp", "atype": "location_recurrent"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"etype": "vggblstmp", "atype": "multi_head_dot"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"etype": "vggblstmp", "atype": "multi_head_add"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"etype": "vggblstmp", "atype": "multi_head_loc"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"etype": "vggblstmp", "atype": "multi_head_multi_res_loc"},
        ),
        ("espnet.nets.pytorch_backend.e2e_st", {"asr_weight": 0.0}),
        ("espnet.nets.pytorch_backend.e2e_st", {"asr_weight": 0.2}),
        ("espnet.nets.pytorch_backend.e2e_st", {"mt_weight": 0.0}),
        ("espnet.nets.pytorch_backend.e2e_st", {"mt_weight": 0.2}),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"asr_weight": 0.2, "mtlalpha": 0.0, "mt_weight": 0.2},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"asr_weight": 0.2, "mtlalpha": 0.5, "mt_weight": 0.2},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"asr_weight": 0.2, "mtlalpha": 1.0, "mt_weight": 0.2},
        ),
        ("espnet.nets.pytorch_backend.e2e_st", {"sampling_probability": 0.5}),
        ("espnet.nets.pytorch_backend.e2e_st", {"context_residual": True}),
        ("espnet.nets.pytorch_backend.e2e_st", {"grad_noise": True}),
        ("espnet.nets.pytorch_backend.e2e_st", {"report_cer": True, "asr_weight": 0.0}),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"report_cer": True, "asr_weight": 0.5, "mtlalpha": 0.0},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"report_cer": True, "asr_weight": 0.5, "mtlalpha": 0.5},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"report_cer": True, "asr_weight": 0.5, "mtlalpha": 1.0},
        ),
        ("espnet.nets.pytorch_backend.e2e_st", {"report_wer": True, "asr_weight": 0.0}),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"report_wer": True, "asr_weight": 0.5, "mtlalpha": 0.0},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"report_wer": True, "asr_weight": 0.5, "mtlalpha": 0.5},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"report_wer": True, "asr_weight": 0.5, "mtlalpha": 1.0},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"report_cer": True, "report_wer": True},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {"report_cer": True, "report_wer": True, "asr_weight": 0.0},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {
                "report_cer": True,
                "report_wer": True,
                "asr_weight": 0.5,
                "mtlalpha": 0.0,
            },
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {
                "report_cer": True,
                "report_wer": True,
                "asr_weight": 0.5,
                "mtlalpha": 0.5,
            },
        ),
        (
            "espnet.nets.pytorch_backend.e2e_st",
            {
                "report_cer": True,
                "report_wer": True,
                "asr_weight": 0.5,
                "mtlalpha": 1.0,
            },
        ),
    ],
)
def test_model_trainable_and_decodable(module, model_dict):
    args = make_arg(**model_dict)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch")
    else:
        raise NotImplementedError

    m = importlib.import_module(module)
    model = m.E2E(40, 5, args)
    loss = model(*batch)
    if isinstance(loss, tuple):
        # chainer return several values as tuple
        loss[0].backward()  # trainable
    else:
        loss.backward()  # trainable

    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(10, 40)
        model.translate(in_data, args, args.char_list)  # decodable
        if "pytorch" in module:
            batch_in_data = [np.random.randn(10, 40), np.random.randn(5, 40)]
            model.translate_batch(
                batch_in_data, args, args.char_list
            )  # batch decodable


@pytest.mark.parametrize("module", ["pytorch"])
def test_gradient_noise_injection(module):
    args = make_arg(grad_noise=True)
    args_org = make_arg()
    dummy_json = make_dummy_json_st(2, [10, 20], [10, 20], [10, 20], idim=20, odim=5)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_st as m
    else:
        raise NotImplementedError
    batchset = make_batchset(dummy_json, 2, 2**10, 2**10, shortest_first=True)
    model = m.E2E(20, 5, args)
    model_org = m.E2E(20, 5, args_org)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=20, odim=5))
        loss_org = model_org(*convert_batch(batch, module, idim=20, odim=5))
        loss.backward()
        grad = [param.grad for param in model.parameters()][10]
        loss_org.backward()
        grad_org = [param.grad for param in model_org.parameters()][10]
        assert grad[0] != grad_org[0]


@pytest.mark.parametrize("module", ["pytorch"])
def test_sortagrad_trainable(module):
    args = make_arg(sortagrad=1)
    dummy_json = make_dummy_json_st(4, [10, 20], [10, 20], [10, 20], idim=20, odim=5)
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_st as m
    else:
        raise NotImplementedError
    batchset = make_batchset(dummy_json, 2, 2**10, 2**10, shortest_first=True)
    model = m.E2E(20, 5, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=20, odim=5))
        if isinstance(loss, tuple):
            # chainer return several values as tuple
            loss[0].backward()  # trainable
        else:
            loss.backward()  # trainable
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(50, 20)
        model.translate(in_data, args, args.char_list)


@pytest.mark.parametrize("module", ["pytorch"])
def test_sortagrad_trainable_with_batch_bins(module):
    args = make_arg(sortagrad=1)
    idim = 20
    odim = 5
    dummy_json = make_dummy_json_st(
        4, [10, 20], [10, 20], [10, 20], idim=idim, odim=odim
    )
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_st as m
    else:
        raise NotImplementedError
    batch_elems = 2000
    batchset = make_batchset(dummy_json, batch_bins=batch_elems, shortest_first=True)
    for batch in batchset:
        n = 0
        for uttid, info in batch:
            ilen = int(info["input"][0]["shape"][0])
            olen = int(info["output"][0]["shape"][0])
            n += ilen * idim + olen * odim
        assert olen < batch_elems

    model = m.E2E(20, 5, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=20, odim=5))
        if isinstance(loss, tuple):
            # chainer return several values as tuple
            loss[0].backward()  # trainable
        else:
            loss.backward()  # trainable
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(100, 20)
        model.translate(in_data, args, args.char_list)


@pytest.mark.parametrize("module", ["pytorch"])
def test_sortagrad_trainable_with_batch_frames(module):
    args = make_arg(sortagrad=1)
    idim = 20
    odim = 5
    dummy_json = make_dummy_json_st(
        4, [10, 20], [10, 20], [10, 20], idim=idim, odim=odim
    )
    if module == "pytorch":
        import espnet.nets.pytorch_backend.e2e_st as m
    else:
        raise NotImplementedError
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

    model = m.E2E(20, 5, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=20, odim=5))
        if isinstance(loss, tuple):
            # chainer return several values as tuple
            loss[0].backward()  # trainable
        else:
            loss.backward()  # trainable
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(100, 20)
        model.translate(in_data, args, args.char_list)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        if p.dim() > 1:
            p.data.fill_(val)


def init_chainer_weight_const(m, val):
    for p in m.params():
        if p.data.ndim > 1:
            p.data[:] = val


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_mtl_loss(etype):
    th = importlib.import_module("espnet.nets.pytorch_backend.e2e_st")
    args = make_arg(etype=etype)
    th_model = th.E2E(40, 5, args)

    const = 1e-4
    init_torch_weight_const(th_model, const)

    th_batch = prepare_inputs("pytorch")

    th_model(*th_batch)
    th_asr, th_st = th_model.loss_asr, th_model.loss_st

    # test grads in mtl mode
    th_loss = th_asr * 0.5 + th_st * 0.5
    th_model.zero_grad()
    th_loss.backward()


@pytest.mark.parametrize("etype", ["blstmp", "vggblstmp"])
def test_zero_length_target(etype):
    th = importlib.import_module("espnet.nets.pytorch_backend.e2e_st")
    args = make_arg(etype=etype)
    th_model = th.E2E(40, 5, args)

    th_batch = prepare_inputs("pytorch", olens_tgt=[4, 0], olens_src=[3, 0])

    th_model(*th_batch)

    # NOTE: We ignore all zero length case because chainer also fails.
    # Have a nice data-prep!
    # out_data = ""
    # data = [
    #     ("aaa", dict(feat=np.random.randn(200, 40).astype(np.float32), tokenid="")),
    #     ("bbb", dict(feat=np.random.randn(100, 40).astype(np.float32), tokenid="")),
    #     ("cc", dict(feat=np.random.randn(100, 40).astype(np.float32), tokenid=""))
    # ]
    # th_asr, th_st, th_acc = th_model(data)


@pytest.mark.parametrize(
    "module, atype",
    [
        ("espnet.nets.pytorch_backend.e2e_st", "noatt"),
        ("espnet.nets.pytorch_backend.e2e_st", "dot"),
        ("espnet.nets.pytorch_backend.e2e_st", "add"),
        ("espnet.nets.pytorch_backend.e2e_st", "location"),
        ("espnet.nets.pytorch_backend.e2e_st", "coverage"),
        ("espnet.nets.pytorch_backend.e2e_st", "coverage_location"),
        ("espnet.nets.pytorch_backend.e2e_st", "location2d"),
        ("espnet.nets.pytorch_backend.e2e_st", "location_recurrent"),
        ("espnet.nets.pytorch_backend.e2e_st", "multi_head_dot"),
        ("espnet.nets.pytorch_backend.e2e_st", "multi_head_add"),
        ("espnet.nets.pytorch_backend.e2e_st", "multi_head_loc"),
        ("espnet.nets.pytorch_backend.e2e_st", "multi_head_multi_res_loc"),
    ],
)
def test_calculate_all_attentions(module, atype):
    m = importlib.import_module(module)
    args = make_arg(atype=atype)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch")
    else:
        raise NotImplementedError
    model = m.E2E(40, 5, args)
    with chainer.no_backprop_mode():
        if "pytorch" in module:
            att_ws = model.calculate_all_attentions(*batch)[0]
        else:
            raise NotImplementedError
        print(att_ws.shape)


@pytest.mark.parametrize(
    "module, mtlalpha",
    [
        ("espnet.nets.pytorch_backend.e2e_st", 0.0),
        ("espnet.nets.pytorch_backend.e2e_st", 0.5),
        ("espnet.nets.pytorch_backend.e2e_st", 1.0),
    ],
)
def test_calculate_all_ctc_probs(module, mtlalpha):
    m = importlib.import_module(module)
    args = make_arg(mtlalpha=mtlalpha, asr_weight=0.3)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch")
    else:
        batch = prepare_inputs("chainer")
    model = m.E2E(40, 5, args)
    with chainer.no_backprop_mode():
        if "pytorch" in module:
            ctc_probs = model.calculate_all_ctc_probs(*batch)
            if mtlalpha > 0:
                print(ctc_probs.shape)
            else:
                assert ctc_probs is None
        else:
            raise NotImplementedError


def test_torch_save_and_load():
    m = importlib.import_module("espnet.nets.pytorch_backend.e2e_st")
    utils = importlib.import_module("espnet.asr.asr_utils")
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


@pytest.mark.skipif(
    not torch.cuda.is_available() and not chainer.cuda.available, reason="gpu required"
)
@pytest.mark.parametrize("module", ["espnet.nets.pytorch_backend.e2e_st"])
def test_gpu_trainable(module):
    m = importlib.import_module(module)
    args = make_arg()
    model = m.E2E(40, 5, args)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
    else:
        raise NotImplementedError
    loss = model(*batch)
    if isinstance(loss, tuple):
        # chainer return several values as tuple
        loss[0].backward()  # trainable
    else:
        loss.backward()  # trainable


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize("module", ["espnet.nets.pytorch_backend.e2e_st"])
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
        loss = 1.0 / ngpu * model(*batch)
        loss.backward(loss.new_ones(ngpu))  # trainable
    else:
        raise NotImplementedError
