# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test VITS related modules."""

import pytest
import torch

from espnet2.tts.vits.vits import VITS
from espnet2.tts.vits.vits import VITSGenerator


def make_vits_generator_args(**kwargs):
    defaults = dict(
        idim=10,
        odim=-1,
        aux_channels=5,
        hidden_channels=4,
        spks=-1,
        global_channels=-1,
        segment_size=4,
        text_encoder_attention_heads=2,
        text_encoder_attention_expand=4,
        text_encoder_blocks=2,
        text_encoder_kernel_size=3,
        text_encoder_dropout_rate=0.1,
        text_encoder_positional_dropout_rate=0.0,
        text_encoder_attention_dropout_rate=0.0,
        decoder_kernel_size=7,
        decoder_channels=16,
        decoder_upsample_scales=(16, 16),
        decoder_upsample_kernel_sizes=(32, 32),
        decoder_resblock_kernel_sizes=(
            3,
            5,
        ),
        decoder_resblock_dilations=[(1, 3), (1, 3)],
        use_weight_norm_in_decoder=True,
        posterior_encoder_kernel_size=5,
        posterior_encoder_layers=2,
        posterior_encoder_stacks=1,
        posterior_encoder_base_dilation=1,
        posterior_encoder_dropout_rate=0.0,
        use_weight_norm_in_posterior_encoder=True,
        flow_flows=2,
        flow_kernel_size=5,
        flow_base_dilation=1,
        flow_layers=2,
        flow_dropout_rate=0.0,
        use_weight_norm_in_flow=True,
        use_only_mean_in_flow=True,
        stochastic_duration_predictor_kernel_size=3,
        stochastic_duration_predictor_dropout_rate=0.5,
        stochastic_duration_predictor_flows=2,
        stochastic_duration_predictor_dds_conv_layers=3,
    )
    defaults.update(kwargs)
    return defaults


def make_vits_discriminator_args(**kwargs):
    defaults = dict(
        scales=2,
        scale_downsample_pooling="AvgPool1d",
        scale_downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 16,
            "max_downsample_channels": 32,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=True,
        periods=[2, 3, 5],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 4,
            "downsample_scales": [3, 3, 1],
            "max_downsample_channels": 16,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    )
    defaults.update(kwargs)
    return defaults


def make_vits_loss_args(**kwargs):
    defaults = dict(
        lambda_adv=1.0,
        lambda_mel=45.0,
        lambda_feat_match=1.0,
        lambda_dur=1.0,
        lambda_kl=1.0,
        generator_adv_loss_params={
            "average_by_discriminators": False,
        },
        discriminator_adv_loss_params={
            "average_by_discriminators": False,
        },
        feat_match_loss_params={
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        mel_loss_params={
            "fs": 22050,
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": None,
            "window": "hann",
            "n_mels": 80,
            "fmin": 0,
            "fmax": None,
            "log_base": None,
        },
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "model_dict",
    [
        ({}),
    ],
)
@torch.no_grad()
def test_vits_generator_forward(model_dict):
    idim = 10
    aux_channels = 5
    args = make_vits_generator_args(idim=idim, aux_channels=aux_channels, **model_dict)
    model = VITSGenerator(**args)

    # check forward
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, aux_channels, 16),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
    )
    outputs = model(**inputs)
    for i, output in enumerate(outputs):
        if not isinstance(output, tuple):
            print(f"{i+1}: {output.shape}")
        else:
            for j, output_ in enumerate(output):
                print(f"{i+j+1}: {output_.shape}")

    # check inference
    inputs = dict(
        text=torch.randint(
            0,
            idim,
            (
                2,
                5,
            ),
        ),
        text_lengths=torch.tensor([5, 3], dtype=torch.long),
    )
    outputs = model.inference(**inputs)
    for i, output in enumerate(outputs):
        if not isinstance(output, tuple):
            print(f"{i+1}: {output.shape}")
        else:
            for j, output_ in enumerate(output):
                print(f"{i+j+1}: {output_.shape}")

    # check inference with teacher forcing
    inputs = dict(
        text=torch.randint(
            0,
            idim,
            (
                1,
                5,
            ),
        ),
        text_lengths=torch.tensor([5], dtype=torch.long),
        dur=torch.tensor([[[1, 2, 3, 4, 5]]], dtype=torch.long),
    )
    outputs = model.inference(**inputs)
    assert outputs[0].size(1) == inputs["dur"].sum() * model.upsample_factor
    for i, output in enumerate(outputs):
        if not isinstance(output, tuple):
            print(f"{i+1}: {output.shape}")
        else:
            for j, output_ in enumerate(output):
                print(f"{i+j+1}: {output_.shape}")


@pytest.mark.parametrize(
    "gen_dict, dis_dict, loss_dict",
    [
        ({}, {}, {}),
    ],
)
def test_vits_is_trainable_and_decodable(gen_dict, dis_dict, loss_dict):
    idim = 10
    aux_channels = 5
    gen_args = make_vits_generator_args(
        idim=idim, aux_channels=aux_channels, **gen_dict
    )
    dis_args = make_vits_discriminator_args(**dis_dict)
    loss_args = make_vits_loss_args(**loss_dict)
    model = VITS(
        idim=idim,
        odim=-1,
        generator_params=gen_args,
        discriminator_params=dis_args,
        **loss_args,
    )
    model.train()
    upsample_factor = model.generator.upsample_factor
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, aux_channels, 16),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        speech=torch.randn(2, 1, 16 * upsample_factor),
        speech_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
    )
    gen_loss, *_ = model.forward_generator(**inputs)
    gen_loss.backward()
    dis_loss, *_ = model.forward_discrminator(**inputs)
    dis_loss.backward()

    with torch.no_grad():
        model.eval()

        # check inference
        inputs = dict(
            text=torch.randint(
                0,
                idim,
                (5,),
            )
        )
        model.inference(**inputs)

        # check inference with teacher forcing
        inputs = dict(
            text=torch.randint(
                0,
                idim,
                (5,),
            ),
            durations=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        )
        outputs = model.inference(**inputs)
        assert outputs[0].size(0) == inputs["durations"].sum() * upsample_factor
