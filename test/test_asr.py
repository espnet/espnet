# coding: utf-8

# Copyright 2018 Hiroshi Seki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import importlib

import numpy
import pytest


def make_arg(**kwargs):
    defaults = dict(
        elayers=4,
        subsample="1_2_2_1_1",
        etype="blstmp",
        eunits=100,
        eprojs=100,
        dlayers=1,
        dunits=300,
        atype="location",
        aconv_chans=10,
        aconv_filts=100,
        mtlalpha=0.5,
        lsm_type="",
        lsm_weight=0.0,
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


def init_torch_weight_const(m, val):
    for p in m.parameters():
        if p.dim() > 1:
            p.data.fill_(val)


def init_chainer_weight_const(m, val):
    for p in m.params():
        if p.data.ndim > 1:
            p.data[:] = val


@pytest.mark.parametrize(("etype", "m_str"), [
    ("blstmp", "e2e_asr_attctc"),
    ("blstmp", "e2e_asr_attctc_th"),
    ("vggblstmp", "e2e_asr_attctc"),
    ("vggblstmp", "e2e_asr_attctc_th"),
])
def test_decoding_results(etype, m_str):
    const = 1e-4
    numpy.random.seed(1)

    args = make_arg(etype=etype)
    m = importlib.import_module(m_str)
    model = m.Loss(m.E2E(40, 5, args), 0.5)

    # (etype, m_str)-dependent
    if "_th" in m_str:
        init_torch_weight_const(model, const)
        seq_true_text = u"ういういういういういういういういお"
    else:
        init_chainer_weight_const(model, const)
        seq_true_text = u"いういういういういういういういうお"

    data = [
        ("aaa", dict(feat=numpy.random.randn(100, 40).astype(
            numpy.float32), token=seq_true_text))
    ]

    in_data = data[0][1]["feat"]
    nbest_hyps = model.predictor.recognize(in_data, args, args.char_list)
    y_hat = nbest_hyps[0]['yseq'][1:]
    seq_hat = [args.char_list[int(idx)] for idx in y_hat]
    seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
    seq_true_text = data[0][1]["token"]

    seq_hat_id = [args.char_list.index(s) for s in seq_hat_text]
    seq_true_id = [args.char_list.index(s) for s in seq_true_text]

    assert seq_hat_text == seq_true_text
