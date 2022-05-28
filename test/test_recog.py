# coding: utf-8

# Copyright 2018 Hiroshi Seki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

import numpy
import pytest
import torch

import espnet.lm.pytorch_backend.extlm as extlm_pytorch
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.pytorch_backend import e2e_asr


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
        word_list=["<blank>", "<unk>", "ai", "iu", "ue", "eo", "oa", "<eos>"],
        outdir=None,
        ctc_type="warpctc",
        report_cer=False,
        report_wer=False,
        sym_space="<space>",
        sym_blank="<blank>",
        context_residual=False,
        use_frontend=False,
        replace_sos=False,
        tgt_lang=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def init_torch_weight_const(m, val):
    for p in m.parameters():
        p.data.fill_(val)


def init_torch_weight_random(m, rand_range):
    for name, p in m.named_parameters():
        p.data.uniform_(rand_range[0], rand_range[1])
        # set small bias for <blank> output
        if "wordlm.lo.bias" in name or "dec.output.bias" in name:
            p.data[0] = -10.0


def init_chainer_weight_const(m, val):
    for p in m.params():
        p.data[:] = val


def make_small_arg(**kwargs):
    return make_arg(
        elayers=1,
        subsample="1_1",
        etype="lstm",
        eunits=2,
        eprojs=2,
        dtype="lstm",
        dlayers=1,
        dunits=2,
        atype="dot",
        adim=2,
        rnnlm="dummy",
        lm_weight=0.3,
        **kwargs
    )


# ctc_weight: 0.0 (attention), 0.5 (hybrid CTC/attention), 1.0 (CTC)
@pytest.mark.parametrize("ctc_weight", [0.0, 0.5, 1.0])
def test_batch_beam_search(ctc_weight):
    numpy.random.seed(1)
    idim = 10
    args = make_small_arg(ctc_weight=ctc_weight)
    model = e2e_asr.E2E(idim, 5, args)
    torch.manual_seed(1)
    rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(len(args.char_list), 2, 2))
    init_torch_weight_random(model, (-0.1, 0.1))
    init_torch_weight_random(rnnlm, (-0.1, 0.1))
    model.eval()
    rnnlm.eval()

    data = [("aaa", dict(feat=numpy.random.randn(10, idim).astype(numpy.float32)))]
    in_data = data[0][1]["feat"]

    s_nbest_hyps = model.recognize(in_data, args, args.char_list)
    b_nbest_hyps = model.recognize_batch([in_data], args, args.char_list)
    assert s_nbest_hyps[0]["yseq"] == b_nbest_hyps[0][0]["yseq"]
    s_nbest_hyps = model.recognize(in_data, args, args.char_list, rnnlm)
    b_nbest_hyps = model.recognize_batch([in_data], args, args.char_list, rnnlm)
    assert s_nbest_hyps[0]["yseq"] == b_nbest_hyps[0][0]["yseq"]

    if ctc_weight > 0.0:
        args.ctc_window_margin = 10
        s_nbest_hyps = model.recognize(in_data, args, args.char_list, rnnlm)
        b_nbest_hyps = model.recognize_batch([in_data], args, args.char_list, rnnlm)
        assert s_nbest_hyps[0]["yseq"] == b_nbest_hyps[0][0]["yseq"]

    # Test word LM in batch decoding
    rand_range = (-0.01, 0.01)
    torch.manual_seed(1)
    char_list = ["<blank>", "<space>"] + args.char_list + ["<eos>"]
    args = make_small_arg(
        ctc_weight=ctc_weight,
        ctc_window_margin=10,
        beam_size=5,
    )
    model = e2e_asr.E2E(idim, len(char_list), args)

    char_dict = {x: i for i, x in enumerate(char_list)}
    word_dict = {x: i for i, x in enumerate(args.word_list)}

    word_rnnlm = lm_pytorch.ClassifierWithState(
        lm_pytorch.RNNLM(len(args.word_list), 2, 2)
    )
    rnnlm = lm_pytorch.ClassifierWithState(
        extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor, word_dict, char_dict)
    )
    init_torch_weight_random(model, rand_range)
    init_torch_weight_random(rnnlm, rand_range)
    model.eval()
    rnnlm.eval()
    s_nbest_hyps = model.recognize(in_data, args, char_list, rnnlm)
    b_nbest_hyps = model.recognize_batch([in_data], args, char_list, rnnlm)
    assert s_nbest_hyps[0]["yseq"] == b_nbest_hyps[0][0]["yseq"]
