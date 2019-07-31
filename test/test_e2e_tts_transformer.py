#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import pytest
import torch

from argparse import Namespace

from espnet.nets.pytorch_backend.e2e_tts_transformer import subsequent_mask
from espnet.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet.nets.pytorch_backend.nets_utils import pad_list


def make_transformer_args(**kwargs):
    defaults = dict(
        embed_dim=512,
        spk_embed_dim=None,
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


def make_inference_args(**kwargs):
    defaults = dict(
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0
    )
    defaults.update(kwargs)
    return defaults


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


@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"use_masking": False}),
        ({"spk_embed_dim": 128}),
        ({"use_scaled_pos_enc": False}),
        ({"bce_pos_weight": 10.0}),
        ({"reduction_factor": 2}),
        ({"reduction_factor": 3}),
        ({"encoder_normalize_before": False}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"encoder_concat_after": True}),
        ({"decoder_concat_after": True}),
        ({"encoder_concat_after": True, "decoder_concat_after": True}),
        ({"loss_type": "L1"}),
        ({"loss_type": "L2"}),
        ({"loss_type": "L1+L2"}),
        ({"use_guided_attn_loss": True}),
        ({"use_guided_attn_loss": True, "modules_applied_guided_attn": ["encoder-decoder"]}),
        ({"use_guided_attn_loss": True, "modules_applied_guided_attn": ["encoder", "decoder"]}),
        ({"use_guided_attn_loss": True, "num_heads_applied_guided_attn": -1}),
        ({"use_guided_attn_loss": True, "num_layers_applied_guided_attn": -1}),
        ({"use_guided_attn_loss": True, "modules_applied_guided_attn": ["encoder"], "elayers": 2, "dlayers": 3}),
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
    batch = prepare_inputs(idim, odim, ilens, olens, model_args["spk_embed_dim"])

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
        ({"spk_embed_dim": 128}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"bce_pos_weight": 10.0}),
        ({"encoder_normalize_before": False}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"decoder_concat_after": True}),
        ({"encoder_concat_after": True, "decoder_concat_after": True}),
    ])
def test_transformer_gpu_trainable(model_dict):
    # make args
    model_args = make_transformer_args(**model_dict)

    idim = 5
    odim = 10
    ilens = [10, 5]
    olens = [20, 15]
    device = torch.device('cuda')
    batch = prepare_inputs(idim, odim, ilens, olens, model_args["spk_embed_dim"], device=device)

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
        ({"spk_embed_dim": 128}),
        ({"use_masking": False}),
        ({"use_scaled_pos_enc": False}),
        ({"bce_pos_weight": 10.0}),
        ({"encoder_normalize_before": False}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"decoder_concat_after": True}),
        ({"encoder_concat_after": True, "decoder_concat_after": True}),
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
    batch = prepare_inputs(idim, odim, ilens, olens, model_args["spk_embed_dim"], device=device)

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


@pytest.mark.parametrize(
    "model_dict", [
        ({}),
        ({"reduction_factor": 3}),
        ({"reduction_factor": 4}),
        ({"decoder_normalize_before": False}),
        ({"encoder_normalize_before": False, "decoder_normalize_before": False}),
        ({"decoder_concat_after": True}),
        ({"encoder_concat_after": True, "decoder_concat_after": True}),
    ])
def test_forward_and_inference_are_equal(model_dict):
    # make args
    model_args = make_transformer_args(dprenet_dropout_rate=0.0, **model_dict)

    # setup batch
    idim = 5
    odim = 10
    ilens = [10]
    olens = [20]
    batch = prepare_inputs(idim, odim, ilens, olens)
    xs = batch["xs"]
    ilens = batch["ilens"]
    ys = batch["ys"]
    olens = batch["olens"]

    # define model
    model = Transformer(idim, odim, Namespace(**model_args))
    model.eval()

    # TODO(kan-bayashi): update following ugly part
    with torch.no_grad():
        # --------- forward calculation ---------
        x_masks = model._source_mask(ilens)
        hs_fp, _ = model.encoder(xs, x_masks)
        if model.reduction_factor > 1:
            ys_in = ys[:, model.reduction_factor - 1::model.reduction_factor]
            olens_in = olens.new([olen // model.reduction_factor for olen in olens])
        else:
            ys_in, olens_in = ys, olens
        ys_in = model._add_first_frame_and_remove_last_frame(ys_in)
        y_masks = model._target_mask(olens_in)
        xy_masks = model._source_to_target_mask(ilens, olens_in)
        zs, _ = model.decoder(ys_in, y_masks, hs_fp, xy_masks)
        before_outs = model.feat_out(zs).view(zs.size(0), -1, model.odim)
        logits = model.prob_out(zs).view(zs.size(0), -1)
        after_outs = before_outs + model.postnet(before_outs.transpose(1, 2)).transpose(1, 2)
        # --------- forward calculation ---------

        # --------- inference calculation ---------
        hs_ir, _ = model.encoder(xs, None)
        maxlen = ys_in.shape[1]
        minlen = ys_in.shape[1]
        idx = 0
        # this is the inferene calculation but we use groundtruth to check the behavior
        ys_in_ = ys_in[0, idx].view(1, 1, model.odim)
        np.testing.assert_array_equal(
            ys_in_.new_zeros(1, 1, model.odim).detach().cpu().numpy(),
            ys_in_.detach().cpu().numpy(),
        )
        outs, probs = [], []
        while True:
            idx += 1
            y_masks = subsequent_mask(idx).unsqueeze(0)
            z = model.decoder.recognize(ys_in_, y_masks, hs_ir)  # (B, idx, adim)
            outs += [model.feat_out(z).view(1, -1, model.odim)]  # [(1, r, odim), ...]
            probs += [torch.sigmoid(model.prob_out(z))[0]]  # [(r), ...]
            if idx >= maxlen:
                if idx < minlen:
                    continue
                outs = torch.cat(outs, dim=1).transpose(1, 2)  # (1, L, odim) -> (1, odim, L)
                if model.postnet is not None:
                    outs = outs + model.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                break
            ys_in_ = torch.cat((ys_in_, ys_in[0, idx].view(1, 1, model.odim)), dim=1)  # (1, idx + 1, odim)
        # --------- inference calculation ---------

        # check both are equal
        np.testing.assert_array_almost_equal(
            hs_fp.detach().cpu().numpy(),
            hs_ir.detach().cpu().numpy(),
        )
        np.testing.assert_array_almost_equal(
            after_outs.squeeze(0).detach().cpu().numpy(),
            outs.detach().cpu().numpy(),
        )
        np.testing.assert_array_almost_equal(
            torch.sigmoid(logits.squeeze(0)).detach().cpu().numpy(),
            probs.detach().cpu().numpy(),
        )
