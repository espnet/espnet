# coding: utf-8

# Copyright 2018 Hiroshi Seki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import importlib
import numpy
import torch

import pytest


def make_arg(**kwargs):
    defaults = dict(
        elayers_sd=1,
        elayers=2,
        subsample="1_2_2_1_1",
        etype="vggblstmp",
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
        verbose=2,
        char_list=["a", "i", "u", "e", "o"],
        outdir=None,
        ctc_type="warpctc",
        num_spkrs=1,
        context_residual=False,
        spa=False
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        p.data.fill_(val)


def init_chainer_weight_const(m, val):
    for p in m.params():
        p.data[:] = val


@pytest.mark.parametrize(("etype", "dtype", "num_spkrs", "spa", "m_str", "text_idx1"), [
    ("vggblstmp", "lstm", 2, True, "espnet.nets.pytorch_backend.e2e_asr_mix", 0),
    ("vggbgrup", "gru", 2, True, "espnet.nets.pytorch_backend.e2e_asr_mix", 1),
])
def test_recognition_results_multi_outputs(etype, dtype, num_spkrs, spa, m_str, text_idx1):
    const = 1e-4
    numpy.random.seed(1)
    seq_true_texts = ([["uiuiuiuiuiuiuiuio", "uiuiuiuiuiuiuiuio"],
                       ["uiuiuiuiuiuiuiuio", "uiuiuiuiuiuiuiuio"]])

    # ctc_weight: 0.5 (hybrid CTC/attention), cannot be 0.0 (attention) or 1.0 (CTC)
    for text_idx2, ctc_weight in enumerate([0.5]):
        seq_true_text_sd = seq_true_texts[text_idx1]

        args = make_arg(etype=etype, ctc_weight=ctc_weight, num_spkrs=num_spkrs, spa=spa)
        m = importlib.import_module(m_str)
        model = m.E2E(40, 5, args)

        if "pytorch" in m_str:
            init_torch_weight_const(model, const)
        else:
            init_chainer_weight_const(model, const)

        data = [
            ("aaa", dict(feat=numpy.random.randn(100, 40).astype(
                numpy.float32), token=seq_true_text_sd))
        ]

        in_data = data[0][1]["feat"]
        nbest_hyps = model.recognize(in_data, args, args.char_list)

        seq_hat_text_sd = []
        for i in range(num_spkrs):
            y_hat = nbest_hyps[i][0]['yseq'][1:]
            seq_hat = [args.char_list[int(idx)] for idx in y_hat]
            seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
            seq_hat_text_sd.append(seq_hat_text)

        seq_true_text_sd = data[0][1]["token"]

        assert seq_hat_text_sd == seq_true_text_sd


@pytest.mark.parametrize(("etype", "dtype", "num_spkrs", "m_str", "data_idx"), [
    ("vggblstmp", "lstm", 2, "espnet.nets.pytorch_backend.e2e_asr_mix", 0),
])
def test_pit_process(etype, dtype, num_spkrs, m_str, data_idx):
    bs = 10
    m = importlib.import_module(m_str)

    losses_2 = torch.ones([bs, 4], dtype=torch.float32)
    for i in range(bs):
        losses_2[i][i % 4] = 0
    true_losses_2 = torch.ones(bs, dtype=torch.float32) / 2
    perm_choices_2 = [[0, 1], [1, 0], [1, 0], [0, 1]]
    true_perm_2 = []
    for i in range(bs):
        true_perm_2.append(perm_choices_2[i % 4])
    true_perm_2 = torch.tensor(true_perm_2).long()

    losses = [losses_2]
    true_losses = [torch.mean(true_losses_2)]
    true_perm = [true_perm_2]

    args = make_arg(etype=etype, num_spkrs=num_spkrs)
    model = m.E2E(40, 5, args)
    min_loss, min_perm = model.pit.pit_process(losses[data_idx])

    assert min_loss == true_losses[data_idx]
    assert torch.equal(min_perm, true_perm[data_idx])
