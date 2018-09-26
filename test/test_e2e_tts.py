#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import pytest
import torch

from argparse import Namespace

from e2e_asr_th import pad_list
from e2e_tts_th import Tacotron2
from e2e_tts_th import Tacotron2Loss


def make_model_args(**kwargs):
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
        cbhg_gru_units=256
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


def prepare_inputs(bs, idim, odim, maxin_len, maxout_len,
                   spk_embed_dim=None, spc_dim=None):
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
    if spk_embed_dim is not None:
        spembs = torch.from_numpy(np.random.randn(bs, spk_embed_dim)).float()
    else:
        spembs = None
    if spc_dim is not None:
        spcs = [np.random.randn(l, spc_dim) for l in olens]
        spcs = pad_list([torch.from_numpy(spc).float() for spc in spcs], 0)
    else:
        spcs = None

    return xs, ilens, ys, labels, olens, spembs, spcs


@pytest.mark.parametrize(
    "model_dict, loss_dict", [
        ({}, {}),
        ({}, {"use_masking": False}),
        ({}, {"bce_pos_weight": 10.0}),
        ({"atype": "forward"}, {}),
        ({"atype": "forward_ta"}, {}),
        ({"prenet_layers": 0}, {}),
        ({"postnet_layers": 0}, {}),
        ({"prenet_layers": 0, "postnet_layers": 0}, {}),
        ({"output_activation": "relu"}, {}),
        ({"cumulate_att_w": False}, {}),
        ({"use_batch_norm": False}, {}),
        ({"use_concate": False}, {}),
        ({"dropout": 0.0}, {}),
        ({"zoneout": 0.0}, {}),
        ({"reduction_factor": 3}, {}),
        ({"use_speaker_embedding": True}, {}),
        ({"use_cbhg": True}, {}),
    ])
def test_tacotron2_trainable_and_decodable(model_dict, loss_dict):
    # make args
    model_args = make_model_args(**model_dict)
    loss_args = make_loss_args(**loss_dict)
    inference_args = make_inference_args()

    # setup batch
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    if model_args['use_cbhg']:
        model_args['spc_dim'] = 129
    if model_args['use_speaker_embedding']:
        model_args['spk_embed_dim'] = 128
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len,
                           model_args['spk_embed_dim'], model_args['spc_dim'])
    xs, ilens, ys, labels, olens, spembs, spcs = batch

    # define model
    model = Tacotron2(idim, odim, Namespace(**model_args))
    criterion = Tacotron2Loss(model, **loss_args)
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = criterion(xs, ilens, ys, labels, olens, spembs, spcs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # decodable
    model.eval()
    with torch.no_grad():
        spemb = None if model_args['spk_embed_dim'] is None else spembs[0]
        model.inference(xs[0][:ilens[0]], Namespace(**inference_args), spemb)
        att_ws = model.calculate_all_attentions(xs, ilens, ys, spembs)
    assert att_ws.shape[0] == bs
    assert att_ws.shape[1] == max(olens)
    assert att_ws.shape[2] == max(ilens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_speaker_embedding": True, "spk_embed_dim": 128}),
        ({"use_cbhg": True, "spc_dim": 128}),
        ({"reduction_factor": 3}),
    ])
def test_tacotron2_gpu_trainable(model_dict):
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    model_args = make_model_args(**model_dict)
    loss_args = make_loss_args()
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len,
                           model_args['spk_embed_dim'], model_args['spc_dim'])
    batch = (x.cuda() if x is not None else None for x in batch)

    # define model
    tacotron2 = Tacotron2(idim, odim, Namespace(**model_args))
    model = Tacotron2Loss(tacotron2, **loss_args)
    optimizer = torch.optim.Adam(model.parameters())
    model.cuda()

    # trainable
    loss = model(*batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_speaker_embedding": True, "spk_embed_dim": 128}),
        ({"use_cbhg": True, "spc_dim": 128}),
        ({"reduction_factor": 3}),
    ])
def test_tacotron2_multi_gpu_trainable(model_dict):
    ngpu = 2
    device_ids = list(range(ngpu))
    bs = 10
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    model_args = make_model_args(**model_dict)
    loss_args = make_loss_args()
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len,
                           model_args['spk_embed_dim'], model_args['spc_dim'])
    batch = (x.cuda() if x is not None else None for x in batch)

    # define model
    tacotron2 = Tacotron2(idim, odim, Namespace(**model_args))
    tacotron2 = torch.nn.DataParallel(tacotron2, device_ids)
    model = Tacotron2Loss(tacotron2, **loss_args)
    optimizer = torch.optim.Adam(model.parameters())
    model.cuda()

    # trainable
    loss = model(*batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
