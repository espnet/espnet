#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import os
import shutil
import tempfile

from argparse import Namespace

import numpy as np
import pytest
import torch

from espnet.nets.pytorch_backend.e2e_tts_fastspeech import FeedForwardTransformer
from espnet.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet.nets.pytorch_backend.fastspeech.duration_calculator import DurationCalculator
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import pad_list


def prepare_inputs(idim, odim, ilens, olens, spk_embed_dim=None,
                   device=torch.device('cpu')):
    ilens = torch.LongTensor(ilens).to(device)
    olens = torch.LongTensor(olens).to(device)
    xs = [np.random.randint(0, idim, l) for l in ilens]
    ys = [np.random.randn(l, odim) for l in olens]
    xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
    ys = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)
    labels = ys.new_zeros(ys.size(0), ys.size(1))
    for i, l in enumerate(olens):
        labels[i, l - 1:] = 1
    batch = {
        "xs": xs,
        "ilens": ilens,
        "ys": ys,
        "labels": labels,
        "olens": olens,
    }

    if spk_embed_dim is not None:
        batch["spembs"] = torch.FloatTensor(np.random.randn(len(ilens), spk_embed_dim)).to(device)

    return batch


def make_transformer_args(**kwargs):
    defaults = dict(
        embed_dim=0,
        spk_embed_dim=None,
        eprenet_conv_layers=0,
        eprenet_conv_filts=0,
        eprenet_conv_chans=0,
        dprenet_layers=2,
        dprenet_units=256,
        adim=32,
        aheads=4,
        elayers=2,
        eunits=128,
        dlayers=2,
        dunits=128,
        postnet_layers=5,
        postnet_filts=5,
        postnet_chans=512,
        eprenet_dropout_rate=0.1,
        dprenet_dropout_rate=0.5,
        postnet_dropout_rate=0.1,
        transformer_enc_dropout_rate=0.1,
        transformer_enc_positional_dropout_rate=0.1,
        transformer_enc_attn_dropout_rate=0.0,
        transformer_dec_dropout_rate=0.1,
        transformer_dec_positional_dropout_rate=0.1,
        transformer_dec_attn_dropout_rate=0.3,
        transformer_enc_dec_attn_dropout_rate=0.0,
        use_masking=True,
        bce_pos_weight=1.0,
        use_batch_norm=True,
        use_scaled_pos_enc=True,
        encoder_normalize_before=True,
        decoder_normalize_before=True,
        encoder_concat_after=False,
        decoder_concat_after=False,
        transformer_init="pytorch",
        initial_encoder_alpha=1.0,
        initial_decoder_alpha=1.0,
        reduction_factor=1,
        loss_type="L1",
        use_guided_attn_loss=False,
        num_heads_applied_guided_attn=2,
        num_layers_applied_guided_attn=2,
        guided_attn_loss_sigma=0.4,
        modules_applied_guided_attn=["encoder", "decoder", "encoder-decoder"]
    )
    defaults.update(kwargs)
    return defaults


def make_feedforward_transformer_args(**kwargs):
    defaults = dict(
        spk_embed_dim=None,
        adim=32,
        aheads=4,
        elayers=2,
        eunits=128,
        dlayers=2,
        dunits=128,
        duration_predictor_layers=2,
        duration_predictor_chans=128,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout_rate=0.1,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        transformer_enc_dropout_rate=0.1,
        transformer_enc_positional_dropout_rate=0.1,
        transformer_enc_attn_dropout_rate=0.0,
        transformer_dec_dropout_rate=0.1,
        transformer_dec_positional_dropout_rate=0.1,
        transformer_dec_attn_dropout_rate=0.3,
        transformer_enc_dec_attn_dropout_rate=0.0,
        use_masking=True,
        use_scaled_pos_enc=True,
        encoder_normalize_before=True,
        decoder_normalize_before=True,
        encoder_concat_after=False,
        decoder_concat_after=False,
        transformer_init="pytorch",
        initial_encoder_alpha=1.0,
        initial_decoder_alpha=1.0,
        transfer_encoder_from_teacher=False,
        transferred_encoder_module="all",
        reduction_factor=1,
        teacher_model=None,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"spk_embed_dim": 128}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"positionwise_layer_type": "conv1d", "positionwise_conv_kernel_size": 3}),
        ({"encoder_normalize_before": False}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"encoder_concat_after": True}),
        ({"decoder_concat_after": True}),
        ({"encoder_concat_after": True, "decoder_concat_after": True}),
    ])
def test_fastspeech_trainable_and_decodable(model_dict):
    # make args
    idim, odim = 10, 25
    teacher_model_args = make_transformer_args(**model_dict)
    model_args = make_feedforward_transformer_args(**model_dict)

    # setup batch
    ilens = [10, 5]
    olens = [20, 15]
    batch = prepare_inputs(idim, odim, ilens, olens, model_args["spk_embed_dim"])

    # define teacher model and save it
    teacher_model = Transformer(idim, odim, Namespace(**teacher_model_args))
    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir="/tmp")
    torch.save(teacher_model.state_dict(), tmpdir + "/model.dummy.best")
    with open(tmpdir + "/model.json", 'wb') as f:
        f.write(json.dumps((idim, odim, teacher_model_args),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

    # define model
    model_args["teacher_model"] = tmpdir + "/model.dummy.best"
    model = FeedForwardTransformer(idim, odim, Namespace(**model_args))
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(**batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # decodable
    model.eval()
    with torch.no_grad():
        model.inference(batch["xs"][0][:batch["ilens"][0]])
        model.calculate_all_attentions(**batch)

    # remove tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"spk_embed_dim": 128}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"encoder_normalize_before": False}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"encoder_concat_after": True}),
        ({"decoder_concat_after": True}),
        ({"encoder_concat_after": True, "decoder_concat_after": True}),
    ])
def test_fastspeech_gpu_trainable(model_dict):
    # make args
    idim, odim = 10, 25
    teacher_model_args = make_transformer_args(**model_dict)
    model_args = make_feedforward_transformer_args(**model_dict)

    # setup batch
    ilens = [10, 5]
    olens = [20, 15]
    device = torch.device('cuda')
    batch = prepare_inputs(idim, odim, ilens, olens, model_args["spk_embed_dim"], device=device)

    # define teacher model and save it
    teacher_model = Transformer(idim, odim, Namespace(**teacher_model_args))
    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir="/tmp")
    torch.save(teacher_model.state_dict(), tmpdir + "/model.dummy.best")
    with open(tmpdir + "/model.json", 'wb') as f:
        f.write(json.dumps((idim, odim, teacher_model_args),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

    # define model
    model_args["teacher_model"] = tmpdir + "/model.dummy.best"
    model = FeedForwardTransformer(idim, odim, Namespace(**model_args))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(**batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # remove tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"spk_embed_dim": 128}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"encoder_normalize_before": False}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"encoder_concat_after": True}),
        ({"decoder_concat_after": True}),
        ({"encoder_concat_after": True, "decoder_concat_after": True}),
    ])
def test_fastspeech_multi_gpu_trainable(model_dict):
    # make args
    idim, odim = 10, 25
    teacher_model_args = make_transformer_args(**model_dict)
    model_args = make_feedforward_transformer_args(**model_dict)

    # setup batch
    ilens = [10, 5]
    olens = [20, 15]
    device = torch.device('cuda')
    batch = prepare_inputs(idim, odim, ilens, olens, model_args["spk_embed_dim"], device=device)

    # define teacher model and save it
    teacher_model = Transformer(idim, odim, Namespace(**teacher_model_args))
    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir="/tmp")
    torch.save(teacher_model.state_dict(), tmpdir + "/model.dummy.best")
    with open(tmpdir + "/model.json", 'wb') as f:
        f.write(json.dumps((idim, odim, teacher_model_args),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

    # define model
    ngpu = 2
    device_ids = list(range(ngpu))
    model_args["teacher_model"] = tmpdir + "/model.dummy.best"
    model = FeedForwardTransformer(idim, odim, Namespace(**model_args))
    model = torch.nn.DataParallel(model, device_ids)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(**batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # remove tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_scaled_pos_enc": False}),
        ({"init_encoder_module": "embed"}),
        ({"encoder_normalize_before": False}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"encoder_concat_after": True}),
        ({"decoder_concat_after": True}),
        ({"encoder_concat_after": True, "decoder_concat_after": True}),
    ])
def test_initialization(model_dict):
    # make args
    idim, odim = 10, 25
    teacher_model_args = make_transformer_args(**model_dict)
    model_args = make_feedforward_transformer_args(**model_dict)

    # define teacher model and save it
    teacher_model = Transformer(idim, odim, Namespace(**teacher_model_args))
    tmpdir = tempfile.mkdtemp(prefix="tmp_", dir="/tmp")
    torch.save(teacher_model.state_dict(), tmpdir + "/model.dummy.best")
    with open(tmpdir + "/model.json", 'wb') as f:
        f.write(json.dumps((idim, odim, teacher_model_args),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

    # define model
    model_args["teacher_model"] = tmpdir + "/model.dummy.best"
    model_args["transfer_encoder_from_teacher"] = True
    model = FeedForwardTransformer(idim, odim, Namespace(**model_args))

    # check initialization
    if model_args["transferred_encoder_module"] == "all":
        for p1, p2 in zip(model.encoder.parameters(), model.teacher.encoder.parameters()):
            np.testing.assert_array_equal(p1.data.cpu().numpy(), p2.data.cpu().numpy())
    else:
        np.testing.assert_array_equal(
            model.encoder.embed[0].weight.data.cpu().numpy(),
            model.teacher.encoder.embed[0].weight.data.cpu().numpy()
        )

    # remove tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


def test_length_regulator():
    # prepare inputs
    idim = 5
    ilens = [10, 5, 3]
    xs = pad_list([torch.randn((ilen, idim)) for ilen in ilens], 0.0)
    ds = pad_list([torch.arange(ilen) for ilen in ilens], 0)

    # test with non-zero durations
    length_regulator = LengthRegulator()
    xs_expand = length_regulator(xs, ds, ilens)
    assert int(xs_expand.shape[1]) == int(ds.sum(dim=-1).max())

    # test with duration including zero
    ds[:, 2] = 0
    xs_expand = length_regulator(xs, ds, ilens)
    assert int(xs_expand.shape[1]) == int(ds.sum(dim=-1).max())


def test_duration_calculator():
    # define duration calculator
    idim, odim = 10, 25
    teacher_model_args = make_transformer_args()
    teacher = Transformer(idim, odim, Namespace(**teacher_model_args))
    duration_calculator = DurationCalculator(teacher)

    # setup batch
    ilens = [10, 5, 3]
    olens = [20, 15, 10]
    batch = prepare_inputs(idim, odim, ilens, olens)

    # calculate durations
    ds = duration_calculator(batch["xs"], batch["ilens"], batch["ys"], batch["olens"])
    np.testing.assert_array_equal(ds.sum(dim=-1).cpu().numpy(), batch["olens"].cpu().numpy())
