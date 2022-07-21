# coding: utf-8

# Copyright 2019 Ruizhi Li
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import importlib
import os
import tempfile
from test.utils_test import make_dummy_json

import numpy as np
import pytest
import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.utils.training.batchfy import make_batchset


def make_arg(num_encs, **kwargs):
    defaults = dict(
        num_encs=num_encs,
        elayers=[1 for _ in range(num_encs)],
        subsample=["1_2_2_1_1" for _ in range(num_encs)],
        etype=["vggblstm" for _ in range(num_encs)],
        eunits=[1 for _ in range(num_encs)],
        eprojs=1,
        dtype="lstm",
        dlayers=1,
        dunits=1,
        atype=["add" for _ in range(num_encs)],
        aheads=[1 for _ in range(num_encs)],
        awin=[1 for _ in range(num_encs)],
        aconv_chans=[1 for _ in range(num_encs)],
        aconv_filts=[1 for _ in range(num_encs)],
        han_type="add",
        han_heads=1,
        han_win=1,
        han_conv_chans=1,
        han_conv_filts=1,
        han_dim=1,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
        sampling_probability=0.0,
        adim=[1 for _ in range(num_encs)],
        dropout_rate=[0.0 for _ in range(num_encs)],
        dropout_rate_decoder=0.0,
        nbest=1,
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
        char_list=["あ", "い"],
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
        share_ctc=False,
        weights_ctc_train=[0.5 for _ in range(num_encs)],
        weights_ctc_dec=[0.5 for _ in range(num_encs)],
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare_inputs(mode, num_encs=2, is_cuda=False):
    ilens_list = [[3, 2] for _ in range(num_encs)]
    olens = [2, 1]
    np.random.seed(1)
    assert len(ilens_list[0]) == len(ilens_list[1]) == len(olens)
    xs_list = [
        [np.random.randn(ilen, 2).astype(np.float32) for ilen in ilens]
        for ilens in ilens_list
    ]
    ys = [np.random.randint(1, 2, olen).astype(np.int32) for olen in olens]
    ilens_list = [np.array([x.shape[0] for x in xs], dtype=np.int32) for xs in xs_list]

    if mode == "pytorch":
        ilens_list = [torch.from_numpy(ilens).long() for ilens in ilens_list]
        xs_pad_list = [
            pad_list([torch.from_numpy(x).float() for x in xs], 0) for xs in xs_list
        ]
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
        if is_cuda:
            xs_pad_list = [xs_pad.cuda() for xs_pad in xs_pad_list]
            ilens_list = [ilens.cuda() for ilens in ilens_list]
            ys_pad = ys_pad.cuda()

        return xs_pad_list, ilens_list, ys_pad
    else:
        raise ValueError("Invalid mode")


def convert_batch(
    batch, backend="pytorch", is_cuda=False, idim=2, odim=2, num_inputs=2
):
    ilens_list = [
        np.array([x[1]["input"][idx]["shape"][0] for x in batch])
        for idx in range(num_inputs)
    ]
    olens = np.array([x[1]["output"][0]["shape"][0] for x in batch])
    xs_list = [
        [np.random.randn(ilen, idim).astype(np.float32) for ilen in ilens_list[idx]]
        for idx in range(num_inputs)
    ]
    ys = [np.random.randint(1, odim, olen).astype(np.int32) for olen in olens]
    is_pytorch = backend == "pytorch"
    if is_pytorch:
        xs_list = [
            pad_list([torch.from_numpy(x).float() for x in xs_list[idx]], 0)
            for idx in range(num_inputs)
        ]
        ilens_list = [
            torch.from_numpy(ilens_list[idx]).long() for idx in range(num_inputs)
        ]
        ys = pad_list([torch.from_numpy(y).long() for y in ys], -1)

        if is_cuda:
            xs_list = [xs_list[idx].cuda() for idx in range(num_inputs)]
            ilens_list = [ilens_list[idx].cuda() for idx in range(num_inputs)]
            ys = ys.cuda()

    return xs_list, ilens_list, ys


@pytest.mark.parametrize(
    "module, num_encs, model_dict",
    [
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {}),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"elayers": [2, 1], "dlayers": 2},
        ),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"etype": ["grup", "grup"]}),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["lstmp", "lstmp"]},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["bgrup", "bgrup"]},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["blstmp", "blstmp"]},
        ),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"etype": ["bgru", "bgru"]}),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["blstm", "blstm"]},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["vgggru", "vgggru"]},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["vgggrup", "vgggrup"]},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["vgglstm", "vgglstm"]},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["vgglstmp", "vgglstmp"]},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["vggbgru", "vggbgru"]},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["vggbgrup", "vggbgrup"]},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"etype": ["blstmp", "vggblstmp"]},
        ),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"dtype": "gru"}),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"atype": ["noatt", "noatt"], "han_type": "noatt"},
        ),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"atype": ["add", "add"]}),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"atype": ["add", "add"]}),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"atype": ["coverage", "coverage"], "han_type": "coverage"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {
                "atype": ["coverage_location", "coverage_location"],
                "han_type": "coverage_location",
            },
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"atype": ["location2d", "location2d"], "han_type": "location2d"},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {
                "atype": ["location_recurrent", "location_recurrent"],
                "han_type": "location_recurrent",
            },
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {
                "atype": ["multi_head_dot", "multi_head_dot"],
                "han_type": "multi_head_dot",
            },
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {
                "atype": ["multi_head_add", "multi_head_add"],
                "han_type": "multi_head_add",
            },
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {
                "atype": ["multi_head_loc", "multi_head_loc"],
                "han_type": "multi_head_loc",
            },
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {
                "atype": ["multi_head_multi_res_loc", "multi_head_multi_res_loc"],
                "han_type": "multi_head_multi_res_loc",
            },
        ),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"mtlalpha": 0.0}),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"mtlalpha": 1.0}),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"sampling_probability": 0.5},
        ),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"ctc_type": "builtin"}),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"ctc_weight": 0.0}),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"ctc_weight": 1.0}),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"context_residual": True}),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"grad_noise": True}),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"report_cer": True}),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"report_wer": True}),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"report_cer": True, "report_wer": True},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"report_cer": True, "report_wer": True, "mtlalpha": 0.0},
        ),
        (
            "espnet.nets.pytorch_backend.e2e_asr_mulenc",
            2,
            {"report_cer": True, "report_wer": True, "mtlalpha": 1.0},
        ),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {"share_ctc": True}),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, {}),
    ],
)
def test_model_trainable_and_decodable(module, num_encs, model_dict):
    args = make_arg(num_encs=num_encs, **model_dict)
    batch = prepare_inputs("pytorch", num_encs)

    # test trainable
    m = importlib.import_module(module)
    model = m.E2E([2 for _ in range(num_encs)], 2, args)
    loss = model(*batch)
    loss.backward()  # trainable

    # test decodable
    with torch.no_grad():
        in_data = [np.random.randn(2, 2) for _ in range(num_encs)]
        model.recognize(in_data, args, args.char_list)  # decodable
        if "pytorch" in module:
            batch_in_data = [
                [np.random.randn(5, 2), np.random.randn(2, 2)] for _ in range(num_encs)
            ]
            model.recognize_batch(
                batch_in_data, args, args.char_list
            )  # batch decodable


@pytest.mark.parametrize("module, num_encs", [("pytorch", 2), ("pytorch", 3)])
def test_gradient_noise_injection(module, num_encs):
    args = make_arg(num_encs=num_encs, grad_noise=True)
    args_org = make_arg(num_encs=num_encs)
    dummy_json = make_dummy_json(
        num_encs, [2, 3], [2, 3], idim=2, odim=2, num_inputs=num_encs
    )
    import espnet.nets.pytorch_backend.e2e_asr_mulenc as m

    batchset = make_batchset(dummy_json, 2, 2 ** 10, 2 ** 10, shortest_first=True)
    model = m.E2E([2 for _ in range(num_encs)], 2, args)
    model_org = m.E2E([2 for _ in range(num_encs)], 2, args_org)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=2, odim=2, num_inputs=num_encs))
        loss_org = model_org(
            *convert_batch(batch, module, idim=2, odim=2, num_inputs=num_encs)
        )
        loss.backward()
        grad = [param.grad for param in model.parameters()][10]
        loss_org.backward()
        grad_org = [param.grad for param in model_org.parameters()][10]
        assert grad[0] != grad_org[0]


@pytest.mark.parametrize("module, num_encs", [("pytorch", 2), ("pytorch", 3)])
def test_sortagrad_trainable(module, num_encs):
    args = make_arg(num_encs=num_encs, sortagrad=1)
    dummy_json = make_dummy_json(6, [2, 3], [2, 3], idim=2, odim=2, num_inputs=num_encs)
    import espnet.nets.pytorch_backend.e2e_asr_mulenc as m

    batchset = make_batchset(dummy_json, 2, 2 ** 10, 2 ** 10, shortest_first=True)
    model = m.E2E([2 for _ in range(num_encs)], 2, args)
    num_utts = 0
    for batch in batchset:
        num_utts += len(batch)
        loss = model(*convert_batch(batch, module, idim=2, odim=2, num_inputs=num_encs))
        loss.backward()  # trainable
    assert num_utts == 6
    with torch.no_grad():
        in_data = [np.random.randn(50, 2) for _ in range(num_encs)]
        model.recognize(in_data, args, args.char_list)


@pytest.mark.parametrize("module, num_encs", [("pytorch", 2), ("pytorch", 3)])
def test_sortagrad_trainable_with_batch_bins(module, num_encs):
    args = make_arg(num_encs=num_encs, sortagrad=1)
    idim = 2
    odim = 2
    dummy_json = make_dummy_json(
        4, [2, 3], [2, 3], idim=idim, odim=odim, num_inputs=num_encs
    )
    import espnet.nets.pytorch_backend.e2e_asr_mulenc as m

    batch_elems = 2000
    batchset = make_batchset(dummy_json, batch_bins=batch_elems, shortest_first=True)
    for batch in batchset:
        n = 0
        for uttid, info in batch:
            ilen = int(info["input"][0]["shape"][0])  # based on the first input
            olen = int(info["output"][0]["shape"][0])
            n += ilen * idim + olen * odim
        assert olen < batch_elems

    model = m.E2E([2 for _ in range(num_encs)], 2, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=2, odim=2, num_inputs=num_encs))
        loss.backward()  # trainable
    with torch.no_grad():
        in_data = [np.random.randn(100, 2) for _ in range(num_encs)]
        model.recognize(in_data, args, args.char_list)


@pytest.mark.parametrize("module, num_encs", [("pytorch", 2), ("pytorch", 3)])
def test_sortagrad_trainable_with_batch_frames(module, num_encs):
    args = make_arg(num_encs=num_encs, sortagrad=1)
    idim = 2
    odim = 2
    dummy_json = make_dummy_json(
        4, [2, 3], [2, 3], idim=idim, odim=odim, num_inputs=num_encs
    )
    import espnet.nets.pytorch_backend.e2e_asr_mulenc as m

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
            i += int(info["input"][0]["shape"][0])  # based on the first input
            o += int(info["output"][0]["shape"][0])
        assert i <= batch_frames_in
        assert o <= batch_frames_out

    model = m.E2E([2 for _ in range(num_encs)], 2, args)
    for batch in batchset:
        loss = model(*convert_batch(batch, module, idim=2, odim=2, num_inputs=num_encs))
        loss.backward()  # trainable
    with torch.no_grad():
        in_data = [np.random.randn(100, 2) for _ in range(num_encs)]
        model.recognize(in_data, args, args.char_list)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        if p.dim() > 1:
            p.data.fill_(val)


@pytest.mark.parametrize(
    "module, num_encs, atype",
    [
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "noatt"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "dot"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "add"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "location"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "coverage"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "coverage_location"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "location2d"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "location_recurrent"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "multi_head_dot"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "multi_head_add"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "multi_head_loc"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, "multi_head_multi_res_loc"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "noatt"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "dot"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "add"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "location"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "coverage"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "coverage_location"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "location2d"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "location_recurrent"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "multi_head_dot"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "multi_head_add"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "multi_head_loc"),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3, "multi_head_multi_res_loc"),
    ],
)
def test_calculate_all_attentions(module, num_encs, atype):
    m = importlib.import_module(module)
    args = make_arg(
        num_encs=num_encs, atype=[atype for _ in range(num_encs)], han_type=atype
    )
    batch = prepare_inputs("pytorch", num_encs)
    model = m.E2E([2 for _ in range(num_encs)], 2, args)
    att_ws = model.calculate_all_attentions(*batch)
    for i in range(num_encs):
        print(att_ws[i][0].shape)  # att
    print(att_ws[num_encs][0].shape)  # han


@pytest.mark.parametrize("num_encs", [2, 3])
def test_torch_save_and_load(num_encs):
    m = importlib.import_module("espnet.nets.pytorch_backend.e2e_asr_mulenc")
    utils = importlib.import_module("espnet.asr.asr_utils")
    args = make_arg(num_encs=num_encs)
    model = m.E2E([2 for _ in range(num_encs)], 2, args)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
@pytest.mark.parametrize(
    "module, num_encs",
    [
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3),
    ],
)
def test_gpu_trainable(module, num_encs):
    m = importlib.import_module(module)
    args = make_arg(num_encs=num_encs)
    model = m.E2E([2 for _ in range(num_encs)], 2, args)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch", num_encs, is_cuda=True)
        model.cuda()
    loss = model(*batch)
    loss.backward()  # trainable


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize(
    "module, num_encs",
    [
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2),
        ("espnet.nets.pytorch_backend.e2e_asr_mulenc", 3),
    ],
)
def test_multi_gpu_trainable(module, num_encs):
    m = importlib.import_module(module)
    ngpu = 2
    device_ids = list(range(ngpu))
    args = make_arg(num_encs=num_encs)
    model = m.E2E([2 for _ in range(num_encs)], 2, args)
    if "pytorch" in module:
        model = torch.nn.DataParallel(model, device_ids)
        batch = prepare_inputs("pytorch", num_encs, is_cuda=True)
        model.cuda()
        loss = 1.0 / ngpu * model(*batch)
        loss.backward(loss.new_ones(ngpu))  # trainable


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize(
    "module, num_encs, model_dict",
    [("espnet.nets.pytorch_backend.e2e_asr_mulenc", 2, {}),],
)
def test_calculate_plot_attention_ctc(module, num_encs, model_dict):
    args = make_arg(num_encs=num_encs, **model_dict)
    m = importlib.import_module(module)
    model = m.E2E([2 for _ in range(num_encs)], 2, args)

    # test attention plot
    dummy_json = make_dummy_json(
        num_encs, [2, 3], [2, 3], idim=2, odim=2, num_inputs=num_encs
    )
    batchset = make_batchset(dummy_json, 2, 2 ** 10, 2 ** 10, shortest_first=True)
    att_ws = model.calculate_all_attentions(
        *convert_batch(batchset[0], "pytorch", idim=2, odim=2, num_inputs=num_encs)
    )
    from espnet.asr.asr_utils import PlotAttentionReport

    tmpdir = tempfile.mkdtemp()
    plot = PlotAttentionReport(
        model.calculate_all_attentions, batchset[0], tmpdir, None, None, None
    )
    for i in range(num_encs):
        # att-encoder
        att_w = plot.trim_attention_weight("utt_%d" % 0, att_ws[i][0])
        plot._plot_and_save_attention(att_w, "{}/att{}.png".format(tmpdir, i))
    # han
    att_w = plot.trim_attention_weight("utt_%d" % 0, att_ws[num_encs][0])
    plot._plot_and_save_attention(att_w, "{}/han.png".format(tmpdir), han_mode=True)

    # test CTC plot
    ctc_probs = model.calculate_all_ctc_probs(
        *convert_batch(batchset[0], "pytorch", idim=2, odim=2, num_inputs=num_encs)
    )
    from espnet.asr.asr_utils import PlotCTCReport

    tmpdir = tempfile.mkdtemp()
    plot = PlotCTCReport(
        model.calculate_all_ctc_probs, batchset[0], tmpdir, None, None, None
    )
    if args.mtlalpha > 0:
        for i in range(num_encs):
            # ctc-encoder
            plot._plot_and_save_ctc(ctc_probs[i][0], "{}/ctc{}.png".format(tmpdir, i))
