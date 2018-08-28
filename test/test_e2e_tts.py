#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import pytest
import torch

from argparse import Namespace

from e2e_asr_th import pad_list
from e2e_tts_th import CBHG
from e2e_tts_th import ConversionLoss
from e2e_tts_th import Tacotron2
from e2e_tts_th import Tacotron2Loss


def make_model_args(**kwargs):
    defaults = dict(
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
        adim=512,
        aconv_chans=32,
        aconv_filts=15,
        cumulate_att_w=True,
        use_batch_norm=True,
        use_concate=True,
        dropout=0.5,
        zoneout=0.1,
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0
    )
    defaults.update(kwargs)
    return defaults


def make_loss_args(**kwargs):
    defaults = dict(
        use_masking=True,
        bce_pos_weight=1.0
    )
    defaults.update(kwargs)
    return defaults


def make_inference_args(**kwargs):
    defaults = dict(
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0
    )
    defaults.update(kwargs)
    return defaults


def prepare_inputs(bs, idim, odim, maxin_len, maxout_len):
    ilens = np.sort(np.random.randint(1, maxin_len, bs))[::-1].tolist()
    olens = np.sort(np.random.randint(1, maxout_len, bs))[::-1].tolist()
    ilens = torch.LongTensor(ilens)
    olens = torch.LongTensor(olens)
    xs = [np.random.randint(0, idim, l) for l in ilens]
    ys = [np.random.randn(l, odim) for l in olens]
    xs = pad_list([torch.from_numpy(x).long() for x in xs], 0)
    ys = pad_list([torch.from_numpy(y).float() for y in ys], 0)
    labels = ys.new_zeros(ys.size(0), ys.size(1))
    for i, l in enumerate(olens):
        labels[i, l - 1:] = 1
    return xs, ilens, ys, labels, olens


@pytest.mark.parametrize(
    "model_dict, loss_dict", [
        ({}, {}),
        ({}, {"use_masking": False}),
        ({}, {"bce_pos_weight": 10.0}),
        ({"prenet_layers": 0}, {}),
        ({"postnet_layers": 0}, {}),
        ({"prenet_layers": 0, "postnet_layers": 0}, {}),
        ({"output_activation": "relu"}, {}),
        ({"cumulate_att_w": False}, {}),
        ({"use_batch_norm": False}, {}),
        ({"use_concate": False}, {}),
        ({"dropout": 0.0}, {}),
        ({"zoneout": 0.0}, {}),
        ({"spk_embed_dim": 128}, {}),
    ])
def test_tacotron2_trainable_and_decodable(model_dict, loss_dict):
    # setup batch
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len)
    xs, ilens, ys, labels, olens = batch

    # define model
    model_args = make_model_args(**model_dict)
    loss_args = make_loss_args(**loss_dict)
    inference_args = make_inference_args()
    model = Tacotron2(idim, odim, Namespace(**model_args))
    criterion = Tacotron2Loss(model, **loss_args)
    optimizer = torch.optim.Adam(model.parameters())

    if model_args['spk_embed_dim'] is not None:
        spembs = torch.from_numpy(np.random.randn(bs, model_args['spk_embed_dim'])).float()
    else:
        spembs = None

    # trainable
    loss = criterion(xs, ilens, ys, labels, olens, spembs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # decodable
    model.eval()
    with torch.no_grad():
        spemb = None if model_args['spk_embed_dim'] is None else spembs[0]
        yhat, probs, att_ws = model.inference(xs[0][:ilens[0]], Namespace(**inference_args), spemb)
        att_ws = model.calculate_all_attentions(xs, ilens, ys, spembs)
    assert att_ws.shape[0] == bs
    assert att_ws.shape[1] == max(olens)
    assert att_ws.shape[2] == max(ilens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
def test_tacotron2_gpu_trainable():
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len)
    batch = (x.cuda() for x in batch)
    xs, ilens, ys, labels, olens = batch

    # define model
    model_args = make_model_args()
    loss_args = make_loss_args()
    tacotron2 = Tacotron2(idim, odim, Namespace(**model_args))
    model = Tacotron2Loss(tacotron2, **loss_args)
    optimizer = torch.optim.Adam(model.parameters())
    model.cuda()

    # trainable
    loss = model(xs, ilens, ys, labels, olens)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
def test_tacotron2_multi_gpu_trainable():
    ngpu = 2
    device_ids = list(range(ngpu))
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len)
    batch = (x.cuda() for x in batch)
    xs, ilens, ys, labels, olens = batch

    # define model
    model_args = make_model_args()
    loss_args = make_loss_args()
    tacotron2 = Tacotron2(idim, odim, Namespace(**model_args))
    tacotron2 = torch.nn.DataParallel(tacotron2, device_ids)
    model = Tacotron2Loss(tacotron2, **loss_args)
    optimizer = torch.optim.Adam(model.parameters())
    model.cuda()

    # trainable
    loss = model(xs, ilens, ys, labels, olens)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_cbhg_trainable_and_decodable():
    bs = 2
    maxin_len = 10
    idim = 80
    odim = 256
    ilens = torch.LongTensor(np.sort(np.random.randint(1, maxin_len, bs))[::-1].tolist())
    xs = pad_list([torch.from_numpy(np.random.randn(l, idim)).float() for l in ilens], 0.0)
    ys = pad_list([torch.from_numpy(np.random.randn(l, odim)).float() for l in ilens], 0.0)

    cbhg = CBHG(idim, odim)
    model = ConversionLoss(cbhg)
    optimizer = torch.optim.Adam(model.parameters())
    loss = model(xs, ilens, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        cbhg.inference(xs[0])
