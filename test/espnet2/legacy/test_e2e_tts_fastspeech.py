#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from argparse import Namespace

import numpy as np
import torch

from espnet2.legacy.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet2.legacy.nets.pytorch_backend.fastspeech.duration_calculator import (  # noqa: H301
    DurationCalculator,
)
from espnet2.legacy.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list


def prepare_inputs(
    idim, odim, ilens, olens, spk_embed_dim=None, device=torch.device("cpu")
):
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
        batch["spembs"] = torch.FloatTensor(
            np.random.randn(len(ilens), spk_embed_dim)
        ).to(device)

    return batch


def make_taco2_args(**kwargs):
    defaults = dict(
        model_module="espnet.nets.pytorch_backend.e2e_tts_tacotron2:Tacotron2",
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


def make_transformer_args(**kwargs):
    defaults = dict(
        model_module="espnet.nets.pytorch_backend.e2e_tts_transformer:Transformer",
        embed_dim=0,
        spk_embed_dim=None,
        eprenet_conv_layers=0,
        eprenet_conv_filts=0,
        eprenet_conv_chans=0,
        dprenet_layers=2,
        dprenet_units=64,
        adim=32,
        aheads=4,
        elayers=2,
        eunits=32,
        dlayers=2,
        dunits=32,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        postnet_layers=2,
        postnet_filts=5,
        postnet_chans=32,
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
        spk_embed_integration_type="add",
        use_masking=True,
        use_weighted_masking=False,
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
        modules_applied_guided_attn=["encoder", "decoder", "encoder-decoder"],
    )
    defaults.update(kwargs)
    return defaults


def make_feedforward_transformer_args(**kwargs):
    defaults = dict(
        spk_embed_dim=None,
        adim=32,
        aheads=4,
        elayers=2,
        eunits=32,
        dlayers=2,
        dunits=32,
        duration_predictor_layers=2,
        duration_predictor_chans=64,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout_rate=0.1,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        postnet_layers=0,
        postnet_filts=5,
        postnet_chans=32,
        transformer_enc_dropout_rate=0.1,
        transformer_enc_positional_dropout_rate=0.1,
        transformer_enc_attn_dropout_rate=0.0,
        transformer_dec_dropout_rate=0.1,
        transformer_dec_positional_dropout_rate=0.1,
        transformer_dec_attn_dropout_rate=0.3,
        transformer_enc_dec_attn_dropout_rate=0.0,
        spk_embed_integration_type="add",
        use_masking=True,
        use_weighted_masking=False,
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


def test_length_regulator():
    # prepare inputs
    idim = 5
    ilens = [10, 5, 3]
    xs = pad_list([torch.randn((ilen, idim)) for ilen in ilens], 0.0)
    ds = pad_list([torch.arange(ilen) for ilen in ilens], 0)

    # test with non-zero durations
    length_regulator = LengthRegulator()
    xs_expand = length_regulator(xs, ds)
    assert int(xs_expand.shape[1]) == int(ds.sum(dim=-1).max())

    # test with duration including zero
    ds[:, 2] = 0
    xs_expand = length_regulator(xs, ds)
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
    np.testing.assert_array_equal(
        ds.sum(dim=-1).cpu().numpy(), batch["olens"].cpu().numpy()
    )

