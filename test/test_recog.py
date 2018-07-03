# coding: utf-8

# Copyright 2018 Hiroshi Seki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import lm_pytorch

import lm_chainer

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
        char_list=["a", "i", "u", "e", "o"],
        outdir=None,
        ctc_type="chainer"
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        p.data.fill_(val)


def init_chainer_weight_const(m, val):
    for p in m.params():
        p.data[:] = val


@pytest.mark.parametrize(("etype", "m_str", "text_idx1"), [
    ("blstmp", "e2e_asr_attctc", 0),
    ("blstmp", "e2e_asr_attctc_th", 1),
    ("vggblstmp", "e2e_asr_attctc", 2),
    ("vggblstmp", "e2e_asr_attctc_th", 3),
])
def test_recognition_results(etype, m_str, text_idx1):
    const = 1e-4
    numpy.random.seed(1)
    seq_true_texts = ([["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                       ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"],
                       ["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                       ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"]])

    # ctc_weight: 0.0 (attention), 0.5 (hybrid CTC/attention), 1.0 (CTC)
    for text_idx2, ctc_weight in enumerate([0.0, 0.5, 1.0]):
        seq_true_text = seq_true_texts[text_idx1][text_idx2]

        args = make_arg(etype=etype, ctc_weight=ctc_weight)
        m = importlib.import_module(m_str)
        model = m.Loss(m.E2E(40, 5, args), 0.5)

        if "_th" in m_str:
            init_torch_weight_const(model, const)
        else:
            init_chainer_weight_const(model, const)

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

        assert seq_hat_text == seq_true_text


@pytest.mark.parametrize(("etype", "m_str", "text_idx1"), [
    ("blstmp", "e2e_asr_attctc", 0),
    ("blstmp", "e2e_asr_attctc_th", 1),
    ("vggblstmp", "e2e_asr_attctc", 2),
    ("vggblstmp", "e2e_asr_attctc_th", 3),
])
def test_recognition_results_with_lm(etype, m_str, text_idx1):
    const = 1e-4
    numpy.random.seed(1)
    seq_true_texts = [["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                      ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"],
                      ["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                      ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"]]

    # ctc_weight: 0.0 (attention), 0.5 (hybrid CTC/attention), 1.0 (CTC)
    for text_idx2, ctc_weight in enumerate([0.0, 0.5, 1.0]):
        seq_true_text = seq_true_texts[text_idx1][text_idx2]

        args = make_arg(etype=etype, rnnlm="dummy", ctc_weight=ctc_weight,
                        lm_weight=0.3)
        m = importlib.import_module(m_str)
        model = m.Loss(m.E2E(40, 5, args), 0.5)

        if "_th" in m_str:
            rnnlm = lm_pytorch.ClassifierWithState(
                lm_pytorch.RNNLM(len(args.char_list), 10))
            init_torch_weight_const(model, const)
            init_torch_weight_const(rnnlm, const)
        else:
            rnnlm = lm_chainer.ClassifierWithState(
                lm_chainer.RNNLM(len(args.char_list), 10))
            init_chainer_weight_const(model, const)
            init_chainer_weight_const(rnnlm, const)

        data = [
            ("aaa", dict(feat=numpy.random.randn(100, 40).astype(
                numpy.float32), token=seq_true_text))
        ]

        in_data = data[0][1]["feat"]
        nbest_hyps = model.predictor.recognize(in_data, args, args.char_list, rnnlm)
        y_hat = nbest_hyps[0]['yseq'][1:]
        seq_hat = [args.char_list[int(idx)] for idx in y_hat]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = data[0][1]["token"]

        assert seq_hat_text == seq_true_text
