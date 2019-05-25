#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import torch

from argparse import Namespace

from espnet.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet.nets.pytorch_backend.nets_utils import pad_list


def make_transformer_args(**kwargs):
    defaults = dict(
        embed_dim=512,
        eprenet_conv_layers=2,
        eprenet_conv_filts=5,
        eprenet_conv_chans=512,
        dprenet_layers=2,
        dprenet_units=256,
        adim=32,
        aheads=4,
        elayers=2,
        eunits=512,
        dlayers=2,
        dunits=512,
        postnet_layers=5,
        postnet_filts=5,
        postnet_chans=512,
        dropout_rate=0.1,
        eprenet_dropout_rate=None,
        dprenet_dropout_rate=None,
        postnet_dropout_rate=None,
        transformer_attn_dropout_rate=None,
        use_masking=True,
        bce_pos_weight=5.0,
        use_batch_norm=True,
        use_scaled_pos_enc=True,
        transformer_init="pytorch"
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
                   spk_embed_dim=None, spc_dim=None, device=torch.device('cpu')):
    ilens = np.sort(np.random.randint(1, maxin_len, bs))[::-1].tolist()
    olens = np.sort(np.random.randint(1, maxout_len, bs))[::-1].tolist()
    ilens = torch.LongTensor(ilens).to(device)
    olens = torch.LongTensor(olens).to(device)
    xs = [np.random.randint(0, idim, l) for l in ilens]
    ys = [np.random.randn(l, odim) for l in olens]
    xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
    ys = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)
    labels = ys.new_zeros(ys.size(0), ys.size(1))
    for i, l in enumerate(olens):
        labels[i, l - 1:] = 1
    if spk_embed_dim is not None:
        spembs = torch.from_numpy(np.random.randn(bs, spk_embed_dim)).float().to(device)
    else:
        spembs = None
    if spc_dim is not None:
        spcs = [np.random.randn(l, spc_dim) for l in olens]
        spcs = pad_list([torch.from_numpy(spc).float() for spc in spcs], 0).to(device)
    else:
        spcs = None

    return xs, ilens, ys, labels, olens, spembs, spcs


@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"bce_pos_weight": 10.0}),
    ])
def test_transformer_trainable_and_decodable(model_dict):
    # make args
    model_args = make_transformer_args(**model_dict)
    inference_args = make_inference_args()

    # setup batch
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len)
    xs, ilens, ys, labels, olens, spembs, spcs = batch

    # define model
    model = Transformer(idim, odim, Namespace(**model_args))
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(xs, ilens, ys, labels, olens).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # check gradient of ScaledPositionalEncoding
    if model.use_scaled_pos_enc:
        assert model.encoder.embed[1].alpha.grad is not None
        assert model.decoder.embed[1].alpha.grad is not None

    # decodable
    model.eval()
    with torch.no_grad():
        model.inference(xs[0][:ilens[0]], Namespace(**inference_args))
        model.calculate_all_attentions(xs, ilens, ys, olens)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"bce_pos_weight": 10.0}),
    ])
def test_transformer_gpu_trainable(model_dict):
    # make args
    model_args = make_transformer_args(**model_dict)

    # setup batch
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    device = torch.device('cuda')
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len, device=device)

    # define model
    model = Transformer(idim, odim, Namespace(**model_args))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(*batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # check gradient of ScaledPositionalEncoding
    if model.use_scaled_pos_enc:
        assert model.encoder.embed[1].alpha.grad is not None
        assert model.decoder.embed[1].alpha.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"bce_pos_weight": 10.0}),
    ])
def test_transformer_multi_gpu_trainable(model_dict):
    # make args
    model_args = make_transformer_args(**model_dict)

    ngpu = 2
    device_ids = list(range(ngpu))

    # setup batch
    bs = 4
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    device = torch.device('cuda')
    batch = prepare_inputs(bs, idim, odim, maxin_len, maxout_len, device=device)

    # define model
    model = Transformer(idim, odim, Namespace(**model_args))
    model = torch.nn.DataParallel(model, device_ids)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(*batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # check gradient of ScaledPositionalEncoding
    if model.module.use_scaled_pos_enc:
        assert model.module.encoder.embed[1].alpha.grad is not None
        assert model.module.decoder.embed[1].alpha.grad is not None
