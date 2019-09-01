# coding: utf-8

# Copyright 2019 Ruizhi Li
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
from test.utils_test import make_dummy_json


def make_arg(**kwargs):
    defaults = dict(
        num_encs=2,
        elayers=[1,1],
        subsample=["1_2_2_1_1", "1_2_2_1_1"],
        etype=["vggblstmp","vggblstmp"],
        eunits=[16,16],
        eprojs=8,
        dtype="lstm",
        dlayers=1,
        dunits=16,
        atype=["location", "location"],
        aheads=[2,2],
        awin=[5, 5],
        aconv_chans=[4, 4],
        aconv_filts=[10, 10],
        han_type="multi_head_add",
        han_heads=2,
        han_win=5,
        han_conv_chans=4,
        han_conv_filts=10,
        han_dim=16,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=[16,16],
        dropout_rate=[0.0, 0.0],
        dropout_rate_decoder=0.0,
        nbest=5,
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
        char_list=[u"あ", u"い", u"う", u"え", u"お"],
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
        replace_sos=False,
        tgt_lang=False,
        share_ctc=False,
        weights_ctc_train=[0.5,0.5],
        weights_ctc_dec=[0.5,0.5],
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare_inputs(mode, ilens_list=[[20, 15], [19, 16]], olens=[4, 3], is_cuda=False):
    np.random.seed(1)
    assert len(ilens_list[0]) == len(ilens_list[1]) == len(olens)
    xs_list = [[np.random.randn(ilen, 40).astype(np.float32) for ilen in ilens] for ilens in ilens_list]
    ys = [np.random.randint(1, 5, olen).astype(np.int32) for olen in olens]
    ilens_list = [np.array([x.shape[0] for x in xs], dtype=np.int32) for xs in xs_list]

    if mode == "pytorch":
        ilens_list = [torch.from_numpy(ilens).long() for ilens in ilens_list]
        xs_pad_list = [pad_list([torch.from_numpy(x).float() for x in xs], 0) for xs in xs_list]
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
        if is_cuda:
            xs_pad_list = [xs_pad.cuda() for xs_pad in xs_pad_list]
            ilens_list = [ilens.cuda() for ilens in ilens_list]
            ys_pad = ys_pad.cuda()

        return xs_pad_list, ilens_list, ys_pad
    else:
        raise ValueError("Invalid mode")


def convert_batch(batch, backend="pytorch", is_cuda=False, idim=40, odim=5, num_inputs=2):
    ilens_list = [np.array([x[1]['input'][idx]['shape'][0] for x in batch]) for idx in range(num_inputs)]
    olens = np.array([x[1]['output'][0]['shape'][0] for x in batch])
    xs_list = [[np.random.randn(ilen, idim).astype(np.float32) for ilen in ilens_list[idx]] for idx in range(num_inputs)]
    ys = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens]
    is_pytorch = backend == "pytorch"
    if is_pytorch:
        xs_list = [pad_list([torch.from_numpy(x).float() for x in xs_list[idx]], 0) for idx in range(num_inputs)]
        ilens_list = [torch.from_numpy(ilens_list[idx]).long() for idx in range(num_inputs)]
        ys = pad_list([torch.from_numpy(y).long() for y in ys], -1)

        if is_cuda:
            xs_list = [xs_list[idx].cuda() for idx in range(num_inputs)]
            ilens_list = [ilens_list[idx].cuda() for idx in range(num_inputs)]
            ys = ys.cuda()

    return xs_list, ilens_list, ys


@pytest.mark.parametrize(
    "module, model_dict", [
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'elayers': [2, 3], 'dlayers': 2}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['grup','grup']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['lstmp','lstmp']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['bgrup','bgrup']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['blstmp','blstmp']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['bgru','bgru']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['blstm','blstm']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['vgggru','vgggru']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['vgggrup','vgggrup']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['vgglstm','vgglstm']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['vgglstmp','vgglstmp']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['vggbgru','vggbgru']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['vggbgrup','vggbgrup']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['vggblstm', 'vggblstm']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'etype': ['blstmp', 'vggblstmp']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'dtype': 'gru'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['noatt','noatt']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['add','add']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['dot','dot']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['coverage','coverage']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['coverage_location','coverage_location']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['location2d','location2d']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['location_recurrent','location_recurrent']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['multi_head_dot','multi_head_dot']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['multi_head_add','multi_head_add']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['multi_head_loc','multi_head_loc']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'atype': ['multi_head_multi_res_loc','multi_head_multi_res_loc']}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'noatt'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'add'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'dot'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'coverage'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'coverage_location'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'location2d'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'location_recurrent'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'multi_head_dot'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'multi_head_add'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'multi_head_loc'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'han_type': 'multi_head_multi_res_loc'}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'mtlalpha': 0.0}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'mtlalpha': 1.0}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'sampling_probability': 0.5}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'ctc_type': "builtin"}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'ctc_weight': 0.0}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'ctc_weight': 1.0}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'context_residual': True}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'grad_noise': True}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'report_cer': True}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'report_wer': True}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'report_cer': True, 'report_wer': True}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'report_cer': True, 'report_wer': True, 'mtlalpha': 0.0}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'report_cer': True, 'report_wer': True, 'mtlalpha': 1.0}),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', {'share_ctc': True})
    ]
)
def test_model_trainable_and_decodable(module, model_dict):
    args = make_arg(**model_dict)
    batch = prepare_inputs("pytorch")

    m = importlib.import_module(module)
    model = m.E2E([40,40], 5, args)
    loss = model(*batch)
    loss.backward()  # trainable

    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = [np.random.randn(10, 40), np.random.randn(9, 40)]
        model.recognize(in_data, args, args.char_list)  # decodable
        if "pytorch" in module:
            batch_in_data = [[np.random.randn(10, 40), np.random.randn(5, 40)], [np.random.randn(9, 40), np.random.randn(4, 40)]]
            model.recognize_batch(batch_in_data, args, args.char_list)  # batch decodable


@pytest.mark.parametrize(
    "module", ["pytorch"]
)
def test_gradient_noise_injection(module):
    args = make_arg(grad_noise=True)
    args_org = make_arg()
    dummy_json = make_dummy_json(2, [10, 20], [10, 20], idim=20, odim=5, num_inputs=2)
    import espnet.nets.pytorch_backend.e2e_asr_mulenc as m
    batchset = make_batchset(dummy_json, 2, 2 ** 10, 2 ** 10, shortest_first=True)
    model = m.E2E([20,20], 5, args)
    model_org = m.E2E([20,20], 5, args_org)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=20, odim=5, num_inputs=2))
        loss_org = model_org(*convert_batch(batch, module, idim=20, odim=5, num_inputs=2))
        loss.backward()
        grad = [param.grad for param in model.parameters()][10]
        loss_org.backward()
        grad_org = [param.grad for param in model_org.parameters()][10]
        assert grad[0] != grad_org[0]


@pytest.mark.parametrize(
    "module", ["pytorch"]
)
def test_sortagrad_trainable(module):
    args = make_arg(sortagrad=1)
    dummy_json = make_dummy_json(4, [10, 20], [10, 20], idim=20, odim=5, num_inputs=2)
    import espnet.nets.pytorch_backend.e2e_asr_mulenc as m
    batchset = make_batchset(dummy_json, 2, 2 ** 10, 2 ** 10, shortest_first=True)
    model = m.E2E([20,20], 5, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=20, odim=5, num_inputs=2))
        loss.backward()  # trainable
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = [np.random.randn(50, 20), np.random.randn(49, 20)]
        model.recognize(in_data, args, args.char_list)


@pytest.mark.parametrize(
    "module", ["pytorch"]
)
def test_sortagrad_trainable_with_batch_bins(module):
    args = make_arg(sortagrad=1)
    idim = 20
    odim = 5
    dummy_json = make_dummy_json(4, [10, 20], [10, 20], idim=idim, odim=odim, num_inputs=2)
    import espnet.nets.pytorch_backend.e2e_asr_mulenc as m
    batch_elems = 2000
    batchset = make_batchset(dummy_json, batch_bins=batch_elems, shortest_first=True)
    for batch in batchset:
        n = 0
        for uttid, info in batch:
            ilen = int(info['input'][0]['shape'][0])  # based on the first input
            olen = int(info['output'][0]['shape'][0])
            n += ilen * idim + olen * odim
        assert olen < batch_elems

    model = m.E2E([20,20], 5, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=20, odim=5, num_inputs=2))
        loss.backward()  # trainable
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = [np.random.randn(100, 20), np.random.randn(99, 20)]
        model.recognize(in_data, args, args.char_list)


@pytest.mark.parametrize(
    "module", ["pytorch"]
)
def test_sortagrad_trainable_with_batch_frames(module):
    args = make_arg(sortagrad=1)
    idim = 20
    odim = 5
    dummy_json = make_dummy_json(4, [10, 20], [10, 20], idim=idim, odim=odim, num_inputs=2)
    import espnet.nets.pytorch_backend.e2e_asr_mulenc as m
    batch_frames_in = 50
    batch_frames_out = 50
    batchset = make_batchset(dummy_json,
                             batch_frames_in=batch_frames_in,
                             batch_frames_out=batch_frames_out,
                             shortest_first=True)
    for batch in batchset:
        i = 0
        o = 0
        for uttid, info in batch:
            i += int(info['input'][0]['shape'][0])   # based on the first input
            o += int(info['output'][0]['shape'][0])
        assert i <= batch_frames_in
        assert o <= batch_frames_out

    model = m.E2E([20,20], 5, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=20, odim=5, num_inputs=2))
        loss.backward()  # trainable
    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = [np.random.randn(100, 20), np.random.randn(99, 20)]
        model.recognize(in_data, args, args.char_list)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        if p.dim() > 1:
            p.data.fill_(val)


@pytest.mark.parametrize(
    "module, atype", [
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'noatt'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'dot'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'add'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'location'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'coverage'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'coverage_location'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'location2d'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'location_recurrent'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'multi_head_dot'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'multi_head_add'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'multi_head_loc'),
        ('espnet.nets.pytorch_backend.e2e_asr_mulenc', 'multi_head_multi_res_loc')
    ]
)
def test_calculate_all_attentions(module, atype):
    m = importlib.import_module(module)
    args = make_arg(atype=[atype, atype], han_type=atype)
    batch = prepare_inputs("pytorch")
    model = m.E2E([40, 40], 5, args)
    with chainer.no_backprop_mode():
        att_ws = model.calculate_all_attentions(*batch)
        print(att_ws[0][0].shape) # att0
        print(att_ws[1][0].shape) # att1
        print(att_ws[2][0].shape) # han


def test_torch_save_and_load():
    m = importlib.import_module('espnet.nets.pytorch_backend.e2e_asr_mulenc')
    utils = importlib.import_module('espnet.asr.asr_utils')
    args = make_arg()
    model = m.E2E([40,40], 5, args)
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
@pytest.mark.parametrize("module", ["espnet.nets.pytorch_backend.e2e_asr_mulenc"])
def test_gpu_trainable(module):
    m = importlib.import_module(module)
    args = make_arg()
    model = m.E2E([40,40], 5, args)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
    loss = model(*batch)
    loss.backward()  # trainable


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize("module", ["espnet.nets.pytorch_backend.e2e_asr_mulenc"])
def test_multi_gpu_trainable(module):
    m = importlib.import_module(module)
    ngpu = 2
    device_ids = list(range(ngpu))
    args = make_arg()
    model = m.E2E([40,40], 5, args)
    if "pytorch" in module:
        model = torch.nn.DataParallel(model, device_ids)
        batch = prepare_inputs("pytorch", is_cuda=True)
        model.cuda()
        loss = 1. / ngpu * model(*batch)
        loss.backward(loss.new_ones(ngpu))  # trainable
