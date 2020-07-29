#!/usr/bin/env python3

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import division

import numpy as np
import pytest
import torch

from argparse import Namespace

from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2
from espnet.nets.pytorch_backend.nets_utils import pad_list


def make_taco2_args(**kwargs):
    defaults = dict(
        use_speaker_embedding=False,
        spk_embed_dim=None,
        embed_dim=32,
        elayers=1,
        eunits=32,
        econv_layers=2,
        econv_filts=5,
        econv_chans=32,
        dlayers=2,
        dunits=32,
        prenet_layers=2,
        prenet_units=32,
        postnet_layers=2,
        postnet_filts=5,
        postnet_chans=32,
        output_activation=None,
        atype="location",
        adim=32,
        aconv_chans=16,
        aconv_filts=5,
        cumulate_att_w=True,
        use_batch_norm=True,
        use_concate=True,
        use_residual=False,
        dropout_rate=0.5,
        zoneout_rate=0.1,
        reduction_factor=1,
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0,
        use_cbhg=False,
        spc_dim=None,
        cbhg_conv_bank_layers=4,
        cbhg_conv_bank_chans=32,
        cbhg_conv_proj_filts=3,
        cbhg_conv_proj_chans=32,
        cbhg_highway_layers=4,
        cbhg_highway_units=32,
        cbhg_gru_units=32,
        use_masking=True,
        use_weighted_masking=False,
        bce_pos_weight=1.0,
        use_guided_attn_loss=False,
        guided_attn_loss_sigma=0.4,
        guided_attn_loss_lambda=1.0,
    )
    defaults.update(kwargs)
    return defaults


def make_inference_args(**kwargs):
    defaults = dict(
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
    )
    defaults.update(kwargs)
    return defaults


def prepare_inputs(
    bs,
    idim,
    odim,
    maxin_len,
    maxout_len,
    spk_embed_dim=None,
    spc_dim=None,
    device=torch.device("cpu"),
):
    ilens = np.sort(np.random.randint(1, maxin_len, bs))[::-1].tolist()
    olens = np.sort(np.random.randint(3, maxout_len, bs))[::-1].tolist()
    xs = [np.random.randint(0, idim, lg) for lg in ilens]
    ys = [np.random.randn(lg, odim) for lg in olens]
    ilens = torch.LongTensor(ilens).to(device)
    olens = torch.LongTensor(olens).to(device)
    xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
    ys = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)
    labels = ys.new_zeros(ys.size(0), ys.size(1))
    for i, lg in enumerate(olens):
        labels[i, lg - 1 :] = 1

    batch = {
        "xs": xs,
        "ilens": ilens,
        "ys": ys,
        "labels": labels,
        "olens": olens,
    }

    if spk_embed_dim is not None:
        spembs = torch.from_numpy(np.random.randn(bs, spk_embed_dim)).float().to(device)
        batch["spembs"] = spembs
    if spc_dim is not None:
        spcs = [np.random.randn(lg, spc_dim) for lg in olens]
        spcs = pad_list([torch.from_numpy(spc).float() for spc in spcs], 0).to(device)
        batch["extras"] = spcs

    return batch


@pytest.mark.parametrize(
    "model_dict, inference_dict",
    [
        ({}, {}),
        ({"use_masking": False}, {}),
        ({"bce_pos_weight": 10.0}, {}),
        ({"atype": "forward"}, {}),
        ({"atype": "forward_ta"}, {}),
        ({"prenet_layers": 0}, {}),
        ({"postnet_layers": 0}, {}),
        ({"prenet_layers": 0, "postnet_layers": 0}, {}),
        ({"output_activation": "relu"}, {}),
        ({"cumulate_att_w": False}, {}),
        ({"use_batch_norm": False}, {}),
        ({"use_concate": False}, {}),
        ({"use_residual": True}, {}),
        ({"dropout_rate": 0.0}, {}),
        ({"zoneout_rate": 0.0}, {}),
        ({"reduction_factor": 2}, {}),
        ({"reduction_factor": 3}, {}),
        ({"use_speaker_embedding": True}, {}),
        ({"use_masking": False}, {}),
        ({"use_masking": False, "use_weighted_masking": True}, {}),
        ({"use_cbhg": True}, {}),
        ({"reduction_factor": 3, "use_cbhg": True}, {}),
        ({"use_guided_attn_loss": True}, {}),
        ({"reduction_factor": 3, "use_guided_attn_loss": True}, {}),
        ({}, {"use_att_constraint": True}),
        ({"atype": "forward"}, {"use_att_constraint": True}),
        ({"atype": "forward_ta"}, {"use_att_constraint": True}),
    ],
)
def test_tacotron2_trainable_and_decodable(model_dict, inference_dict):
    # make args
    model_args = make_taco2_args(**model_dict)
    inference_args = make_inference_args(**inference_dict)

    # setup batch
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    if model_args["use_cbhg"]:
        model_args["spc_dim"] = 129
    if model_args["use_speaker_embedding"]:
        model_args["spk_embed_dim"] = 128
    batch = prepare_inputs(
        bs,
        idim,
        odim,
        maxin_len,
        maxout_len,
        model_args["spk_embed_dim"],
        model_args["spc_dim"],
    )

    # define model
    model = Tacotron2(idim, odim, Namespace(**model_args))
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(**batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # decodable
    model.eval()
    with torch.no_grad():
        spemb = None if model_args["spk_embed_dim"] is None else batch["spembs"][0]
        model.inference(
            batch["xs"][0][: batch["ilens"][0]], Namespace(**inference_args), spemb
        )
        model.calculate_all_attentions(**batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
@pytest.mark.parametrize(
    "model_dict, inference_dict",
    [
        ({}, {}),
        ({"atype": "forward"}, {}),
        ({"atype": "forward_ta"}, {}),
        ({"use_speaker_embedding": True, "spk_embed_dim": 128}, {}),
        ({"use_cbhg": True, "spc_dim": 128}, {}),
        ({"reduction_factor": 3}, {}),
        ({"use_guided_attn_loss": True}, {}),
        ({"use_masking": False}, {}),
        ({"use_masking": False, "use_weighted_masking": True}, {}),
        ({}, {"use_att_constraint": True}),
        ({"atype": "forward"}, {"use_att_constraint": True}),
        ({"atype": "forward_ta"}, {"use_att_constraint": True}),
    ],
)
def test_tacotron2_gpu_trainable_and_decodable(model_dict, inference_dict):
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    device = torch.device("cuda")
    model_args = make_taco2_args(**model_dict)
    inference_args = make_inference_args(**inference_dict)
    batch = prepare_inputs(
        bs,
        idim,
        odim,
        maxin_len,
        maxout_len,
        model_args["spk_embed_dim"],
        model_args["spc_dim"],
        device=device,
    )

    # define model
    model = Tacotron2(idim, odim, Namespace(**model_args))
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)

    # trainable
    loss = model(**batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # decodable
    model.eval()
    with torch.no_grad():
        spemb = None if model_args["spk_embed_dim"] is None else batch["spembs"][0]
        model.inference(
            batch["xs"][0][: batch["ilens"][0]], Namespace(**inference_args), spemb
        )
        model.calculate_all_attentions(**batch)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize(
    "model_dict",
    [
        ({}),
        ({"atype": "forward"}),
        ({"atype": "forward_ta"}),
        ({"use_speaker_embedding": True, "spk_embed_dim": 128}),
        ({"use_cbhg": True, "spc_dim": 128}),
        ({"reduction_factor": 3}),
        ({"use_guided_attn_loss": True}),
        ({"use_masking": False}),
        ({"use_masking": False, "use_weighted_masking": True}),
    ],
)
def test_tacotron2_multi_gpu_trainable(model_dict):
    ngpu = 2
    device_ids = list(range(ngpu))
    device = torch.device("cuda")
    bs = 10
    maxin_len = 10
    maxout_len = 10
    idim = 5
    odim = 10
    model_args = make_taco2_args(**model_dict)
    batch = prepare_inputs(
        bs,
        idim,
        odim,
        maxin_len,
        maxout_len,
        model_args["spk_embed_dim"],
        model_args["spc_dim"],
        device=device,
    )

    # define model
    model = Tacotron2(idim, odim, Namespace(**model_args))
    model = torch.nn.DataParallel(model, device_ids)
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)

    # trainable
    loss = model(**batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
