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
        bce_pos_weight=1.0,
        use_batch_norm=True,
        use_scaled_pos_enc=True,
        encoder_normalize_before=True,
        decoder_normalize_before=True,
        encoder_concate_after=False,
        decoder_concate_after=False,
        transformer_init="pytorch",
        initial_encoder_alpha=1.0,
        initial_decoder_alpha=1.0,
        reduction_factor=1,
        loss_type="L1",
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


def prepare_inputs(idim, odim, ilens, olens,
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

    return batch


@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"bce_pos_weight": 10.0}),
        ({"encoder_normalize_before": False}),
        ({"reduction_factor": 2}),
        ({"reduction_factor": 3}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"decoder_concate_after": True}),
        ({"encoder_concate_after": True, "decoder_concate_after": True}),
        ({"loss_type": "L1"}),
        ({"loss_type": "L2"}),
        ({"loss_type": "L1+L2"}),
    ])
def test_transformer_trainable_and_decodable(model_dict):
    # make args
    model_args = make_transformer_args(**model_dict)
    inference_args = make_inference_args()

    # setup batch
    idim = 5
    odim = 10
    ilens = [10, 5]
    olens = [20, 15]
    batch = prepare_inputs(idim, odim, ilens, olens)

    # define model
    model = Transformer(idim, odim, Namespace(**model_args))
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(**batch).mean()
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
        model.inference(batch["xs"][0][:batch["ilens"][0]], Namespace(**inference_args))
        model.calculate_all_attentions(**batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu required")
@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"bce_pos_weight": 10.0}),
        ({"encoder_normalize_before": False}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"decoder_concate_after": True}),
        ({"encoder_concate_after": True, "decoder_concate_after": True}),
    ])
def test_transformer_gpu_trainable(model_dict):
    # make args
    model_args = make_transformer_args(**model_dict)

    idim = 5
    odim = 10
    ilens = [10, 5]
    olens = [20, 15]
    device = torch.device('cuda')
    batch = prepare_inputs(idim, odim, ilens, olens, device=device)

    # define model
    model = Transformer(idim, odim, Namespace(**model_args))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(**batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # check gradient of ScaledPositionalEncoding
    if model.use_scaled_pos_enc:
        assert model.encoder.embed[1].alpha.grad is not None
        assert model.decoder.embed[1].alpha.grad is not None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi gpu required")
@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"bce_pos_weight": 10.0}),
        ({"encoder_normalize_before": False}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"decoder_concate_after": True}),
        ({"encoder_concate_after": True, "decoder_concate_after": True}),
    ])
def test_transformer_multi_gpu_trainable(model_dict):
    # make args
    model_args = make_transformer_args(**model_dict)

    # setup batch
    idim = 5
    odim = 10
    ilens = [10, 5]
    olens = [20, 15]
    device = torch.device('cuda')
    batch = prepare_inputs(idim, odim, ilens, olens, device=device)

    # define model
    ngpu = 2
    device_ids = list(range(ngpu))
    model = Transformer(idim, odim, Namespace(**model_args))
    model = torch.nn.DataParallel(model, device_ids)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = model(**batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # check gradient of ScaledPositionalEncoding
    if model.module.use_scaled_pos_enc:
        assert model.module.encoder.embed[1].alpha.grad is not None
        assert model.module.decoder.embed[1].alpha.grad is not None


@pytest.mark.parametrize(
    "model_dict", [
        ({}),
    ])
def test_attention_masking(model_dict):
    # make args
    model_args = make_transformer_args(**model_dict)

    # setup batch
    idim = 5
    odim = 10
    ilens = [10, 5]
    olens = [20, 15]
    batch = prepare_inputs(idim, odim, ilens, olens)

    # define model
    model = Transformer(idim, odim, Namespace(**model_args))

    # test encoder self-attention
    xs = model.encoder.embed(batch["xs"])
    xs[1, ilens[1]:] = float("nan")
    x_masks = model._source_mask(batch["ilens"])
    a = model.encoder.encoders[0].self_attn
    a(xs, xs, xs, x_masks)
    aws = a.attn.detach().numpy()
    assert not np.isnan(aws).any()
    for aw, ilen in zip(aws, batch["ilens"]):
        np.testing.assert_almost_equal(aw[:, :ilen, :ilen].sum(), float(aw.shape[0] * ilen), decimal=4)
        assert aw[:, ilen:, ilen:].sum() == 0.0

    # test encoder-decoder attention
    ys = model.decoder.embed(batch["ys"])
    ys[1, olens[1]:] = float("nan")
    xy_masks = model._source_to_target_mask(batch["ilens"], batch["olens"])
    a = model.decoder.decoders[0].src_attn
    a(ys, xs, xs, xy_masks)
    aws = a.attn.detach().numpy()
    assert not np.isnan(aws).any()
    for aw, ilen, olen in zip(aws, batch["ilens"], batch["olens"]):
        np.testing.assert_almost_equal(aw[:, :olen, :ilen].sum(), float(aw.shape[0] * olen), decimal=4)
        assert aw[:, olen:, ilen:].sum() == 0.0

    # test decoder self-attention
    ys = model.decoder.embed(batch["ys"])
    ys[1, olens[1]:] = float("nan")
    y_masks = model._target_mask(batch["olens"])
    a = model.decoder.decoders[0].self_attn
    a(ys, ys, ys, y_masks)
    aws = a.attn.detach().numpy()
    assert not np.isnan(aws).any()
    for aw, olen in zip(aws, batch["olens"]):
        np.testing.assert_almost_equal(aw[:, :olen, :olen].sum(), float(aw.shape[0] * olen), decimal=4)
        assert aw[:, olen:, olen:].sum() == 0.0
