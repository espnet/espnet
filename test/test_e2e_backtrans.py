#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import pytest
import torch

from bts_pytorch import pad_ndarray_list
from e2e_asr_backtrans import Tacotron2
from e2e_asr_backtrans import Tacotron2Loss

torch_is_old = torch.__version__.startswith("0.3.")


def make_model_args(**kwargs):
    defaults = dict(
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
        output_activation_fn=None,
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


@pytest.mark.skipif(torch_is_old, reason="torch version is old")
@pytest.mark.parametrize(
    "model_dict, loss_dict", [
        ({}, {}),
        ({}, {"use_masking": False}),
        ({}, {"bce_pos_weight": 10.0}),
        ({"prenet_layers": 0}, {}),
        ({"postnet_layers": 0}, {}),
        ({"prenet_layers": 0, "postnet_layers": 0}, {}),
        ({"output_activation_fn": torch.nn.functional.tanh}, {}),
        ({"cumulate_att_w": False}, {}),
        ({"use_batch_norm": False}, {}),
        ({"use_concate": False}, {}),
        ({"dropout": 0.0}, {}),
        ({"zoneout": 0.0}, {}),
    ])
def test_tacotron2_trainable_and_decodable(model_dict, loss_dict):
    # setup batch
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    ilens = np.sort(np.random.randint(1, maxin_len, bs))[::-1].tolist()
    olens = np.sort(np.random.randint(1, maxout_len, bs))[::-1].tolist()
    xs = pad_ndarray_list([np.random.randint(0, idim, l) for l in ilens], 0)
    ys = pad_ndarray_list([np.random.randn(l, odim) for l in olens], 0)
    xs = torch.from_numpy(xs).long()
    ys = torch.from_numpy(ys).float()
    labels = ys.new_zeros((ys.size(0), ys.size(1)))
    for i, l in enumerate(olens):
        labels[i, l - 1:] = 1

    # define model
    model_args = make_model_args(**model_dict)
    loss_args = make_loss_args(**loss_dict)
    model = Tacotron2(idim, odim, **model_args)
    criterion = Tacotron2Loss(model, **loss_args)
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    after, before, logits = model(xs, ilens, ys)
    loss = criterion(xs, ilens, ys, labels, olens)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # decodable
    with torch.no_grad():
        model.eval()
        yhat, probs, att_ws = model.inference(xs[0][:ilens[0]])
        att_ws = model.calculate_all_attentions(xs, ilens, ys)
        assert att_ws.shape[0] == bs
        assert att_ws.shape[1] == max(olens)
        assert att_ws.shape[2] == max(ilens)
