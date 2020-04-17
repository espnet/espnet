# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import chainer
import importlib

import numpy as np
import pytest
import torch

from espnet.asr.pytorch_backend.asrtts import Reporter
from espnet.nets.pytorch_backend.e2e_asrtts import TacotronRewardLoss
from espnet.nets.pytorch_backend.nets_utils import pad_list


def make_arg(**kwargs):
    defaults = dict(
        elayers=4,
        subsample="1_2_2_1_1",
        etype="vggblstm",
        eunits=100,
        eprojs=100,
        dtype="lstm",
        dlayers=1,
        dunits=300,
        atype="location",
        aheads=4,
        awin=5,
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
        streaming_min_blank_dur=10,
        streaming_onset_margin=2,
        streaming_offset_margin=2,
        verbose=2,
        char_list=[u"あ", u"い", u"う", u"え", u"お"],
        outdir=None,
        ctc_type="warpctc",
        sym_space="<space>",
        sym_blank="<blank>",
        sortagrad=0,
        grad_noise=False,
        context_residual=False,
        use_speaker_embedding=True,
        expected_loss="tts",
        alpha=0.5,
        n_samples_per_input=5,
        policy_gradient=True,
        sample_scaling=0.1,
        generator="tts",
        rnnloss=None,
        update_asr_only=True
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def make_tts_args(**kwargs):
    defaults = dict(
        use_speaker_embedding=False,
        spk_embed_dim=None,
        embed_dim=512,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_filts=5,
        econv_chans=512,
        dlayers=2,
        dunits=1024,
        prenet_layers=2,
        prenet_units=256,
        postnet_layers=5,
        postnet_filts=5,
        postnet_chans=512,
        output_activation=None,
        atype="location",
        adim=512,
        aconv_chans=32,
        aconv_filts=15,
        cumulate_att_w=True,
        use_batch_norm=True,
        use_concate=True,
        dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0,
        use_cbhg=False,
        spc_dim=None,
        cbhg_conv_bank_layers=8,
        cbhg_conv_bank_chans=128,
        cbhg_conv_proj_filts=3,
        cbhg_conv_proj_chans=256,
        cbhg_highway_layers=4,
        cbhg_highway_units=128,
        cbhg_gru_units=256,
        use_masking=True,
        bce_pos_weight=1.0,
        use_guided_attn_loss=False,
        guided_attn_loss_sigma=0.4,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare_inputs(mode, ilens=[150, 100], olens=[4, 3], is_cuda=False):
    np.random.seed(1)
    assert len(ilens) == len(olens)
    xs = [np.random.randn(ilen, 40).astype(np.float32) for ilen in ilens]
    ys = [np.random.randint(1, 5, olen).astype(np.int32) for olen in olens]
    ilens = np.array([x.shape[0] for x in xs], dtype=np.int32)
    embs = torch.from_numpy(np.random.randn(2, 128)).float().to(torch.device('cpu'))

    if mode == "chainer":
        if is_cuda:
            xp = importlib.import_module('cupy')
            xs = [chainer.Variable(xp.array(x)) for x in xs]
            ys = [chainer.Variable(xp.array(y)) for y in ys]
            ilens = xp.array(ilens)
            embs = [chainer.Variable(xp.array(e)) for e in embs]
        else:
            xs = [chainer.Variable(x) for x in xs]
            ys = [chainer.Variable(y) for y in ys]
        return xs, ilens, ys, embs

    elif mode == "pytorch":
        ilens = torch.from_numpy(ilens).long()
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1)
        embs = torch.from_numpy(np.random.randn(2, 128)).float().to(torch.device('cpu'))
        if is_cuda:
            xs_pad = xs_pad.cuda()
            ilens = ilens.cuda()
            ys_pad = ys_pad.cuda()
            embs = embs.cuda()
        return xs_pad, ilens, ys_pad, embs
    else:
        raise ValueError("Invalid mode")


@pytest.mark.parametrize(
    "module, etype, atype, dtype", [
        ('espnet.nets.pytorch_backend.e2e_asrtts', 'blstmp', 'location', 'lstm'),
    ]
)
def test_model_trainable_and_decodable(module, etype, atype, dtype):
    args = make_arg(etype=etype, atype=atype, dtype=dtype)
    if "pytorch" in module:
        batch = prepare_inputs("pytorch", is_cuda=False)
    else:
        batch = prepare_inputs("chainer", is_cuda=False)

    m = importlib.import_module(module)
    tts_args = make_tts_args(spk_embed_dim=128, use_speaker_embedding=True)
    loss_fn = TacotronRewardLoss(None, idim=5, odim=40, train_args=tts_args, reporter=Reporter())
    asr_model = m.E2E(40, 5, args)
    model = m.E2E(40, 5, args, predictor=asr_model, asr2tts=True, loss_fn=loss_fn, rnnlm=None)
    attn_loss = model(*batch)[0]
    attn_loss.backward()  # trainable

    with torch.no_grad(), chainer.no_backprop_mode():
        in_data = np.random.randn(100, 40)
        model.recognize(in_data, args, args.char_list)  # decodable
        if "pytorch" in module:
            batch_in_data = [np.random.randn(100, 40), np.random.randn(50, 40)]
            model.recognize_batch(batch_in_data, args, args.char_list)  # batch decodable


# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
# @pytest.mark.parametrize("module", ["espnet.nets.pytorch_backend.e2e_asrtts"])
# def test_model_trainable_gpu(module):
#    ngpu = 1
#    args = make_arg()
#    m = importlib.import_module(module)
#    device_ids = list(range(ngpu))
#    tts_args = make_tts_args(spk_embed_dim=128, use_speaker_embedding=True)
#    loss_fn = TacotronRewardLoss(None, idim=5, odim=40, train_args=tts_args, reporter=Reporter())
#    asr_model = m.E2E(40, 5, args)
#    model = m.E2E(40, 5, args, predictor=asr_model, asr2tts=True, loss_fn=loss_fn, rnnlm=None)
#    if "pytorch" in module:
#        model = torch.nn.DataParallel(model, device_ids)
#        batch = prepare_inputs("pytorch", is_cuda=True)
#        model.cuda()
#        loss = 1. / ngpu * model(*batch)[0]
#        loss.backward(loss.new_ones(ngpu))  # trainable
