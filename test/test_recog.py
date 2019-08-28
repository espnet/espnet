# coding: utf-8

# Copyright 2018 Hiroshi Seki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import espnet.nets.pytorch_backend.lm.default as lm_pytorch

import espnet.lm.chainer_backend.lm as lm_chainer

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
        dtype="lstm",
        dlayers=1,
        dunits=300,
        atype="location",
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
        ctc_window_margin=0,
        verbose=2,
        char_list=["a", "i", "u", "e", "o"],
        outdir=None,
        ctc_type="warpctc",
        report_cer=False,
        report_wer=False,
        sym_space="<space>",
        sym_blank="<blank>",
        context_residual=False,
        use_frontend=False,
        replace_sos=False,
        tgt_lang=False
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        p.data.fill_(val)


def init_chainer_weight_const(m, val):
    for p in m.params():
        p.data[:] = val


@pytest.mark.parametrize(("etype", "dtype", "m_str", "text_idx1"), [
    ("blstmp", "lstm", "espnet.nets.chainer_backend.e2e_asr", 0),
    ("blstmp", "lstm", "espnet.nets.pytorch_backend.e2e_asr", 1),
    ("vggblstmp", "lstm", "espnet.nets.chainer_backend.e2e_asr", 2),
    ("vggblstmp", "lstm", "espnet.nets.pytorch_backend.e2e_asr", 3),
    ("bgrup", "gru", "espnet.nets.chainer_backend.e2e_asr", 4),
    ("bgrup", "gru", "espnet.nets.pytorch_backend.e2e_asr", 5),
    ("vggbgrup", "gru", "espnet.nets.chainer_backend.e2e_asr", 6),
    ("vggbgrup", "gru", "espnet.nets.pytorch_backend.e2e_asr", 7),
])
def test_recognition_results(etype, dtype, m_str, text_idx1):
    const = 1e-4
    numpy.random.seed(1)
    seq_true_texts = ([["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                       ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"],
                       ["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                       ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"],
                       ["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                       ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"],
                       ["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                       ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"]])

    # ctc_weight: 0.0 (attention), 0.5 (hybrid CTC/attention), 1.0 (CTC)
    for text_idx2, ctc_weight in enumerate([0.0, 0.5, 1.0]):
        seq_true_text = seq_true_texts[text_idx1][text_idx2]

        args = make_arg(etype=etype, ctc_weight=ctc_weight)
        m = importlib.import_module(m_str)
        model = m.E2E(40, 5, args)

        if "pytorch" in m_str:
            init_torch_weight_const(model, const)
        else:
            init_chainer_weight_const(model, const)

        data = [
            ("aaa", dict(feat=numpy.random.randn(100, 40).astype(
                numpy.float32), token=seq_true_text))
        ]

        in_data = data[0][1]["feat"]
        nbest_hyps = model.recognize(in_data, args, args.char_list)
        y_hat = nbest_hyps[0]['yseq'][1:]
        seq_hat = [args.char_list[int(idx)] for idx in y_hat]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = data[0][1]["token"]

        assert seq_hat_text == seq_true_text


@pytest.mark.parametrize(("etype", "dtype", "m_str", "text_idx1"), [
    ("blstmp", "lstm", "espnet.nets.chainer_backend.e2e_asr", 0),
    ("blstmp", "lstm", "espnet.nets.pytorch_backend.e2e_asr", 1),
    ("vggblstmp", "lstm", "espnet.nets.chainer_backend.e2e_asr", 2),
    ("vggblstmp", "lstm", "espnet.nets.pytorch_backend.e2e_asr", 3),
    ("bgrup", "gru", "espnet.nets.chainer_backend.e2e_asr", 4),
    ("bgrup", "gru", "espnet.nets.pytorch_backend.e2e_asr", 5),
    ("vggbgrup", "gru", "espnet.nets.chainer_backend.e2e_asr", 6),
    ("vggbgrup", "gru", "espnet.nets.pytorch_backend.e2e_asr", 7),
])
def test_recognition_results_with_lm(etype, dtype, m_str, text_idx1):
    const = 1e-4
    numpy.random.seed(1)
    seq_true_texts = [["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                      ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"],
                      ["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                      ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"],
                      ["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                      ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"],
                      ["o", "iuiuiuiuiuiuiuiuo", "aiaiaiaiaiaiaiaio"],
                      ["o", "uiuiuiuiuiuiuiuio", "aiaiaiaiaiaiaiaio"]]

    # ctc_weight: 0.0 (attention), 0.5 (hybrid CTC/attention), 1.0 (CTC)
    for text_idx2, ctc_weight in enumerate([0.0, 0.5, 1.0]):
        seq_true_text = seq_true_texts[text_idx1][text_idx2]

        args = make_arg(etype=etype, rnnlm="dummy", ctc_weight=ctc_weight,
                        lm_weight=0.3)
        m = importlib.import_module(m_str)
        model = m.E2E(40, 5, args)

        if "pytorch" in m_str:
            rnnlm = lm_pytorch.ClassifierWithState(
                lm_pytorch.RNNLM(len(args.char_list), 2, 10))
            init_torch_weight_const(model, const)
            init_torch_weight_const(rnnlm, const)
        else:
            rnnlm = lm_chainer.ClassifierWithState(
                lm_chainer.RNNLM(len(args.char_list), 2, 10))
            init_chainer_weight_const(model, const)
            init_chainer_weight_const(rnnlm, const)

        data = [
            ("aaa", dict(feat=numpy.random.randn(100, 40).astype(
                numpy.float32), token=seq_true_text))
        ]

        in_data = data[0][1]["feat"]
        nbest_hyps = model.recognize(in_data, args, args.char_list, rnnlm)
        y_hat = nbest_hyps[0]['yseq'][1:]
        seq_hat = [args.char_list[int(idx)] for idx in y_hat]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = data[0][1]["token"]

        assert seq_hat_text == seq_true_text


@pytest.mark.parametrize(("etype", "dtype", "m_str"), [
    ("blstmp", "lstm", "espnet.nets.chainer_backend.e2e_asr"),
    ("blstmp", "lstm", "espnet.nets.pytorch_backend.e2e_asr"),
    ("vggblstmp", "lstm", "espnet.nets.chainer_backend.e2e_asr"),
    ("vggblstmp", "lstm", "espnet.nets.pytorch_backend.e2e_asr"),
    ("bgrup", "gru", "espnet.nets.chainer_backend.e2e_asr"),
    ("bgrup", "gru", "espnet.nets.pytorch_backend.e2e_asr"),
    ("vggbgrup", "gru", "espnet.nets.chainer_backend.e2e_asr"),
    ("vggbgrup", "gru", "espnet.nets.pytorch_backend.e2e_asr"),
])
def test_batch_beam_search(etype, dtype, m_str):
    const = 1e-4
    numpy.random.seed(1)

    # ctc_weight: 0.0 (attention), 0.5 (hybrid CTC/attention), 1.0 (CTC)
    for ctc_weight in [0.0, 0.5]:
        args = make_arg(etype=etype, rnnlm="dummy", ctc_weight=ctc_weight,
                        lm_weight=0.3)
        m = importlib.import_module(m_str)
        model = m.E2E(40, 5, args)

        if "pytorch" in m_str:
            rnnlm = lm_pytorch.ClassifierWithState(
                lm_pytorch.RNNLM(len(args.char_list), 2, 10))
            init_torch_weight_const(model, const)
            init_torch_weight_const(rnnlm, const)
        else:
            # chainer module
            continue

        data = [("aaa", dict(feat=numpy.random.randn(100, 40).astype(numpy.float32)))]
        in_data = data[0][1]["feat"]

        for lm_weight in [0.0, 0.3]:
            if lm_weight == 0.0:
                s_nbest_hyps = model.recognize(in_data, args, args.char_list)
                b_nbest_hyps = model.recognize_batch([in_data], args, args.char_list)
            else:
                s_nbest_hyps = model.recognize(in_data, args, args.char_list, rnnlm)
                b_nbest_hyps = model.recognize_batch([in_data], args, args.char_list, rnnlm)

            assert s_nbest_hyps[0]['yseq'] == b_nbest_hyps[0][0]['yseq']

        if ctc_weight > 0.0:
            args.ctc_window_margin = 40
            s_nbest_hyps = model.recognize(in_data, args, args.char_list, rnnlm)
            b_nbest_hyps = model.recognize_batch([in_data], args, args.char_list, rnnlm)
            assert s_nbest_hyps[0]['yseq'] == b_nbest_hyps[0][0]['yseq']
