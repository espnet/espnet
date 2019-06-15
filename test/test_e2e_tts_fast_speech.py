#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from argparse import Namespace

import numpy as np
import pytest
import torch

from espnet.nets.pytorch_backend.e2e_tts_fast_speech import DurationCalculator
from espnet.nets.pytorch_backend.e2e_tts_fast_speech import FeedForwardTransformer
from espnet.nets.pytorch_backend.e2e_tts_fast_speech import LengthRegularizer
from espnet.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet.nets.pytorch_backend.nets_utils import pad_list


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
        adim=32,
        aheads=4,
        elayers=2,
        eunits=512,
        dlayers=2,
        dunits=512,
        duration_predictor_layers=2,
        duration_predictor_chans=128,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout_rate=0.1,
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
        reduction_factor=1,
        loss_type="L1",
        teacher_model=None,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "model_dict", [
        ({}),
    ])
def test_trainable_and_decodable(model_dict):
    # make args
    idim, odim = 10, 25
    teacher_model_args = make_transformer_args()
    model_args = make_feedforward_transformer_args(**model_dict)

    # setup batch
    ilens = [10, 5]
    olens = [20, 15]
    batch = prepare_inputs(idim, odim, ilens, olens)

    # define model
    model = FeedForwardTransformer(idim, odim, Namespace(**model_args))
    teacher_model = Transformer(idim, odim, Namespace(**teacher_model_args))
    model.teacher = teacher_model
    model.duration_calculator = DurationCalculator(model.teacher)
    optimizer = torch.optim.Adam(model.parameters())

    # tranable
    loss = model(**batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_length_regularizer():
    # prepare inputs
    idim = 5
    ilens = [10, 5, 3]
    xs = pad_list([torch.randn((ilen, idim)) for ilen in ilens], 0.0)
    ds = pad_list([torch.arange(ilen) for ilen in ilens], 0)

    # test with non-zero durations
    length_regularizer = LengthRegularizer()
    xs_expand = length_regularizer(xs, ds, ilens)
    assert int(xs_expand.shape[1]) == int(ds.sum(dim=-1).max())

    # test with duration including zero
    ds[:, 2] = 0
    xs_expand = length_regularizer(xs, ds, ilens)
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
