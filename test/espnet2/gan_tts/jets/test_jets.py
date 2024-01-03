# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test JETS related modules."""

import pytest
import torch

from espnet2.gan_tts.jets import JETS


def make_jets_generator_args(**kwargs):
    defaults = dict(
        generator_type="jets_generator",
        generator_params={
            "idim": 10,
            "odim": 5,
            "adim": 4,
            "aheads": 2,
            "elayers": 1,
            "eunits": 4,
            "dlayers": 1,
            "dunits": 4,
            "positionwise_layer_type": "conv1d",
            "positionwise_conv_kernel_size": 1,
            "use_scaled_pos_enc": True,
            "use_batch_norm": True,
            "encoder_normalize_before": True,
            "decoder_normalize_before": True,
            "encoder_concat_after": False,
            "decoder_concat_after": False,
            "reduction_factor": 1,
            "encoder_type": "transformer",
            "decoder_type": "transformer",
            "transformer_enc_dropout_rate": 0.1,
            "transformer_enc_positional_dropout_rate": 0.1,
            "transformer_enc_attn_dropout_rate": 0.1,
            "transformer_dec_dropout_rate": 0.1,
            "transformer_dec_positional_dropout_rate": 0.1,
            "transformer_dec_attn_dropout_rate": 0.1,
            "conformer_rel_pos_type": "legacy",
            "conformer_pos_enc_layer_type": "rel_pos",
            "conformer_self_attn_layer_type": "rel_selfattn",
            "conformer_activation_type": "swish",
            "use_macaron_style_in_conformer": True,
            "use_cnn_in_conformer": True,
            "zero_triu": False,
            "conformer_enc_kernel_size": 3,
            "conformer_dec_kernel_size": 3,
            "duration_predictor_layers": 2,
            "duration_predictor_chans": 4,
            "duration_predictor_kernel_size": 3,
            "duration_predictor_dropout_rate": 0.1,
            "energy_predictor_layers": 2,
            "energy_predictor_chans": 4,
            "energy_predictor_kernel_size": 3,
            "energy_predictor_dropout": 0.5,
            "energy_embed_kernel_size": 3,
            "energy_embed_dropout": 0.5,
            "stop_gradient_from_energy_predictor": False,
            "pitch_predictor_layers": 2,
            "pitch_predictor_chans": 4,
            "pitch_predictor_kernel_size": 3,
            "pitch_predictor_dropout": 0.5,
            "pitch_embed_kernel_size": 3,
            "pitch_embed_dropout": 0.5,
            "stop_gradient_from_pitch_predictor": False,
            "spks": None,
            "langs": None,
            "spk_embed_dim": None,
            "spk_embed_integration_type": "add",
            "use_gst": False,
            "gst_tokens": 10,
            "gst_heads": 4,
            "gst_conv_layers": 2,
            "gst_conv_chans_list": (3, 3, 6, 6, 12, 12),
            "gst_conv_kernel_size": 3,
            "gst_conv_stride": 2,
            "gst_gru_layers": 1,
            "gst_gru_units": 8,
            "init_type": "xavier_uniform",
            "init_enc_alpha": 1.0,
            "init_dec_alpha": 1.0,
            "use_masking": False,
            "use_weighted_masking": False,
            "segment_size": 4,
            "generator_out_channels": 1,
            "generator_channels": 16,
            "generator_global_channels": -1,
            "generator_kernel_size": 7,
            "generator_upsample_scales": [16, 16],
            "generator_upsample_kernel_sizes": [32, 32],
            "generator_resblock_kernel_sizes": [3, 3],
            "generator_resblock_dilations": [
                [1, 3],
                [1, 3],
            ],
            "generator_use_additional_convs": True,
            "generator_bias": True,
            "generator_nonlinear_activation": "LeakyReLU",
            "generator_nonlinear_activation_params": {"negative_slope": 0.1},
            "generator_use_weight_norm": True,
        },
    )
    defaults.update(kwargs)
    return defaults


def make_jets_discriminator_args(**kwargs):
    defaults = dict(
        discriminator_type="hifigan_multi_scale_multi_period_discriminator",
        discriminator_params={
            "scales": 1,
            "scale_downsample_pooling": "AvgPool1d",
            "scale_downsample_pooling_params": {
                "kernel_size": 4,
                "stride": 2,
                "padding": 2,
            },
            "scale_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [15, 41, 5, 3],
                "channels": 16,
                "max_downsample_channels": 32,
                "max_groups": 16,
                "bias": True,
                "downsample_scales": [2, 1],
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
            },
            "follow_official_norm": False,
            "periods": [2, 3],
            "period_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 4,
                "downsample_scales": [3, 1],
                "max_downsample_channels": 16,
                "bias": True,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
        },
    )
    defaults.update(kwargs)
    return defaults


def make_jets_loss_args(**kwargs):
    defaults = dict(
        lambda_adv=1.0,
        lambda_mel=45.0,
        lambda_feat_match=2.0,
        lambda_var=1.0,
        lambda_align=2.0,
        generator_adv_loss_params={
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        discriminator_adv_loss_params={
            "average_by_discriminators": False,
            "loss_type": "mse",
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


# NOTE(kan-bayashi): first forward requires jit compile
#   so a little bit more time is needed to run. Therefore,
#   here we extend execution timeout from 2 sec to 8 sec.
@pytest.mark.execution_timeout(8)
@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "gen_dict, dis_dict, loss_dict",
    [
        ({}, {}, {}),
        ({}, {}, {"cache_generator_outputs": True}),
        ({}, {}, {"plot_pred_mos": True}),
        (
            {},
            {
                "discriminator_type": "hifigan_multi_scale_discriminator",
                "discriminator_params": {
                    "scales": 2,
                    "downsample_pooling": "AvgPool1d",
                    "downsample_pooling_params": {
                        "kernel_size": 4,
                        "stride": 2,
                        "padding": 2,
                    },
                    "discriminator_params": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "kernel_sizes": [15, 41, 5, 3],
                        "channels": 16,
                        "max_downsample_channels": 32,
                        "max_groups": 16,
                        "bias": True,
                        "downsample_scales": [2, 2, 1],
                        "nonlinear_activation": "LeakyReLU",
                        "nonlinear_activation_params": {"negative_slope": 0.1},
                    },
                },
            },
            {},
        ),
        (
            {},
            {
                "discriminator_type": "hifigan_multi_period_discriminator",
                "discriminator_params": {
                    "periods": [2, 3],
                    "discriminator_params": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "kernel_sizes": [5, 3],
                        "channels": 16,
                        "downsample_scales": [3, 3, 1],
                        "max_downsample_channels": 32,
                        "bias": True,
                        "nonlinear_activation": "LeakyReLU",
                        "nonlinear_activation_params": {"negative_slope": 0.1},
                        "use_weight_norm": True,
                        "use_spectral_norm": False,
                    },
                },
            },
            {},
        ),
        (
            {},
            {},
            {
                "generator_adv_loss_params": {
                    "average_by_discriminators": True,
                    "loss_type": "mse",
                },
                "discriminator_adv_loss_params": {
                    "average_by_discriminators": True,
                    "loss_type": "mse",
                },
            },
        ),
        (
            {},
            {},
            {
                "generator_adv_loss_params": {
                    "average_by_discriminators": False,
                    "loss_type": "hinge",
                },
                "discriminator_adv_loss_params": {
                    "average_by_discriminators": False,
                    "loss_type": "hinge",
                },
            },
        ),
    ],
)
def test_jets_is_trainable_and_decodable(gen_dict, dis_dict, loss_dict):
    idim = 10
    odim = 5
    gen_args = make_jets_generator_args(**gen_dict)
    dis_args = make_jets_discriminator_args(**dis_dict)
    loss_args = make_jets_loss_args(**loss_dict)
    model = JETS(
        idim=idim,
        odim=odim,
        **gen_args,
        **dis_args,
        **loss_args,
    )
    model.train()
    upsample_factor = model.generator.upsample_factor
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, 16, odim),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        speech=torch.randn(2, 16 * upsample_factor),
        speech_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
        pitch=torch.randn(2, 16, 1),
        pitch_lengths=torch.tensor([16, 13], dtype=torch.long),
        energy=torch.randn(2, 16, 1),
        energy_lengths=torch.tensor([16, 13], dtype=torch.long),
    )
    gen_loss = model(forward_generator=True, **inputs)["loss"]
    gen_loss.backward()
    dis_loss = model(forward_generator=False, **inputs)["loss"]
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

        # check inference with teachder forcing
        inputs = dict(
            text=torch.randint(
                0,
                idim,
                (5,),
            ),
            feats=torch.randn(16, odim),
            pitch=torch.randn(16, 1),
            energy=torch.randn(16, 1),
        )
        output_dict = model.inference(**inputs, use_teacher_forcing=True)
        assert output_dict["wav"].size(0) == inputs["feats"].size(0) * upsample_factor


@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="Group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "gen_dict, dis_dict, loss_dict,",
    [
        ({}, {}, {}),
        ({}, {}, {"cache_generator_outputs": True}),
        (
            {},
            {
                "discriminator_type": "hifigan_multi_scale_discriminator",
                "discriminator_params": {
                    "scales": 2,
                    "downsample_pooling": "AvgPool1d",
                    "downsample_pooling_params": {
                        "kernel_size": 4,
                        "stride": 2,
                        "padding": 2,
                    },
                    "discriminator_params": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "kernel_sizes": [15, 41, 5, 3],
                        "channels": 16,
                        "max_downsample_channels": 32,
                        "max_groups": 16,
                        "bias": True,
                        "downsample_scales": [2, 2, 1],
                        "nonlinear_activation": "LeakyReLU",
                        "nonlinear_activation_params": {"negative_slope": 0.1},
                    },
                },
            },
            {},
        ),
        (
            {},
            {
                "discriminator_type": "hifigan_multi_period_discriminator",
                "discriminator_params": {
                    "periods": [2, 3],
                    "discriminator_params": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "kernel_sizes": [5, 3],
                        "channels": 16,
                        "downsample_scales": [3, 3, 1],
                        "max_downsample_channels": 32,
                        "bias": True,
                        "nonlinear_activation": "LeakyReLU",
                        "nonlinear_activation_params": {"negative_slope": 0.1},
                        "use_weight_norm": True,
                        "use_spectral_norm": False,
                    },
                },
            },
            {},
        ),
        (
            {},
            {},
            {
                "generator_adv_loss_params": {
                    "average_by_discriminators": True,
                    "loss_type": "mse",
                },
                "discriminator_adv_loss_params": {
                    "average_by_discriminators": True,
                    "loss_type": "mse",
                },
            },
        ),
        (
            {},
            {},
            {
                "generator_adv_loss_params": {
                    "average_by_discriminators": False,
                    "loss_type": "hinge",
                },
                "discriminator_adv_loss_params": {
                    "average_by_discriminators": False,
                    "loss_type": "hinge",
                },
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "spks, spk_embed_dim, langs", [(10, -1, -1), (-1, 5, -1), (-1, -1, 3), (4, 5, 3)]
)
def test_multi_speaker_jets_is_trainable_and_decodable(
    gen_dict, dis_dict, loss_dict, spks, spk_embed_dim, langs
):
    idim = 10
    odim = 5
    gen_args = make_jets_generator_args(**gen_dict)
    gen_args["generator_params"]["spks"] = spks
    gen_args["generator_params"]["langs"] = langs
    gen_args["generator_params"]["spk_embed_dim"] = spk_embed_dim
    dis_args = make_jets_discriminator_args(**dis_dict)
    loss_args = make_jets_loss_args(**loss_dict)
    model = JETS(
        idim=idim,
        odim=odim,
        **gen_args,
        **dis_args,
        **loss_args,
    )
    model.train()
    upsample_factor = model.generator.upsample_factor
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, 16, odim),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        speech=torch.randn(2, 16 * upsample_factor),
        speech_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
        pitch=torch.randn(2, 16, 1),
        pitch_lengths=torch.tensor([16, 13], dtype=torch.long),
        energy=torch.randn(2, 16, 1),
        energy_lengths=torch.tensor([16, 13], dtype=torch.long),
    )
    if spks > 0:
        inputs["sids"] = torch.randint(0, spks, (2, 1))
    if langs > 0:
        inputs["lids"] = torch.randint(0, langs, (2, 1))
    if spk_embed_dim > 0:
        inputs["spembs"] = torch.randn(2, spk_embed_dim)
    gen_loss = model(forward_generator=True, **inputs)["loss"]
    gen_loss.backward()
    dis_loss = model(forward_generator=False, **inputs)["loss"]
    dis_loss.backward()

    with torch.no_grad():
        model.eval()

        # check inference
        inputs = dict(
            text=torch.randint(
                0,
                idim,
                (5,),
            ),
        )
        if spks > 0:
            inputs["sids"] = torch.randint(0, spks, (1,))
        if langs > 0:
            inputs["lids"] = torch.randint(0, langs, (1,))
        if spk_embed_dim > 0:
            inputs["spembs"] = torch.randn(spk_embed_dim)
        model.inference(**inputs)

        # check inference with teacher forcing
        inputs = dict(
            text=torch.randint(
                0,
                idim,
                (5,),
            ),
            feats=torch.randn(16, odim),
            pitch=torch.randn(16, 1),
            energy=torch.randn(16, 1),
        )
        if spks > 0:
            inputs["sids"] = torch.randint(0, spks, (1,))
        if langs > 0:
            inputs["lids"] = torch.randint(0, langs, (1,))
        if spk_embed_dim > 0:
            inputs["spembs"] = torch.randn(spk_embed_dim)
        output_dict = model.inference(**inputs, use_teacher_forcing=True)
        assert output_dict["wav"].size(0) == inputs["feats"].size(0) * upsample_factor


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU is needed.",
)
@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "gen_dict, dis_dict, loss_dict",
    [
        ({}, {}, {}),
        ({}, {}, {"cache_generator_outputs": True}),
        (
            {},
            {
                "discriminator_type": "hifigan_multi_scale_discriminator",
                "discriminator_params": {
                    "scales": 2,
                    "downsample_pooling": "AvgPool1d",
                    "downsample_pooling_params": {
                        "kernel_size": 4,
                        "stride": 2,
                        "padding": 2,
                    },
                    "discriminator_params": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "kernel_sizes": [15, 41, 5, 3],
                        "channels": 16,
                        "max_downsample_channels": 32,
                        "max_groups": 16,
                        "bias": True,
                        "downsample_scales": [2, 2, 1],
                        "nonlinear_activation": "LeakyReLU",
                        "nonlinear_activation_params": {"negative_slope": 0.1},
                    },
                },
            },
            {},
        ),
        (
            {},
            {
                "discriminator_type": "hifigan_multi_period_discriminator",
                "discriminator_params": {
                    "periods": [2, 3],
                    "discriminator_params": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "kernel_sizes": [5, 3],
                        "channels": 16,
                        "downsample_scales": [3, 3, 1],
                        "max_downsample_channels": 32,
                        "bias": True,
                        "nonlinear_activation": "LeakyReLU",
                        "nonlinear_activation_params": {"negative_slope": 0.1},
                        "use_weight_norm": True,
                        "use_spectral_norm": False,
                    },
                },
            },
            {},
        ),
        (
            {},
            {
                "discriminator_type": "hifigan_period_discriminator",
                "discriminator_params": {
                    "period": 2,
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [5, 3],
                    "channels": 16,
                    "downsample_scales": [3, 3, 1],
                    "max_downsample_channels": 32,
                    "bias": True,
                    "nonlinear_activation": "LeakyReLU",
                    "nonlinear_activation_params": {"negative_slope": 0.1},
                    "use_weight_norm": True,
                    "use_spectral_norm": False,
                },
            },
            {},
        ),
        (
            {},
            {
                "discriminator_type": "hifigan_scale_discriminator",
                "discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [15, 41, 5, 3],
                    "channels": 16,
                    "max_downsample_channels": 32,
                    "max_groups": 16,
                    "bias": True,
                    "downsample_scales": [2, 2, 1],
                    "nonlinear_activation": "LeakyReLU",
                    "nonlinear_activation_params": {"negative_slope": 0.1},
                },
            },
            {},
        ),
        (
            {},
            {},
            {
                "generator_adv_loss_params": {
                    "average_by_discriminators": True,
                    "loss_type": "mse",
                },
                "discriminator_adv_loss_params": {
                    "average_by_discriminators": True,
                    "loss_type": "mse",
                },
            },
        ),
        (
            {},
            {},
            {
                "generator_adv_loss_params": {
                    "average_by_discriminators": False,
                    "loss_type": "hinge",
                },
                "discriminator_adv_loss_params": {
                    "average_by_discriminators": False,
                    "loss_type": "hinge",
                },
            },
        ),
    ],
)
def test_jets_is_trainable_and_decodable_on_gpu(gen_dict, dis_dict, loss_dict):
    idim = 10
    odim = 5
    gen_args = make_jets_generator_args(**gen_dict)
    dis_args = make_jets_discriminator_args(**dis_dict)
    loss_args = make_jets_loss_args(**loss_dict)
    model = JETS(
        idim=idim,
        odim=odim,
        **gen_args,
        **dis_args,
        **loss_args,
    )
    model.train()
    upsample_factor = model.generator.upsample_factor
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, 16, odim),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        speech=torch.randn(2, 16 * upsample_factor),
        speech_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
        pitch=torch.randn(2, 16, 1),
        pitch_lengths=torch.tensor([16, 13], dtype=torch.long),
        energy=torch.randn(2, 16, 1),
        energy_lengths=torch.tensor([16, 13], dtype=torch.long),
    )
    device = torch.device("cuda")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_loss = model(forward_generator=True, **inputs)["loss"]
    gen_loss.backward()
    dis_loss = model(forward_generator=False, **inputs)["loss"]
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
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.inference(**inputs)

        # check inference with teacher forcing
        inputs = dict(
            text=torch.randint(
                0,
                idim,
                (5,),
            ),
            feats=torch.randn(16, odim),
            pitch=torch.randn(16, 1),
            energy=torch.randn(16, 1),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_dict = model.inference(**inputs, use_teacher_forcing=True)
        assert output_dict["wav"].size(0) == inputs["feats"].size(0) * upsample_factor


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU is needed.",
)
@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="Group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "gen_dict, dis_dict, loss_dict",
    [
        ({}, {}, {}),
        ({}, {}, {"cache_generator_outputs": True}),
        (
            {},
            {
                "discriminator_type": "hifigan_multi_scale_discriminator",
                "discriminator_params": {
                    "scales": 2,
                    "downsample_pooling": "AvgPool1d",
                    "downsample_pooling_params": {
                        "kernel_size": 4,
                        "stride": 2,
                        "padding": 2,
                    },
                    "discriminator_params": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "kernel_sizes": [15, 41, 5, 3],
                        "channels": 16,
                        "max_downsample_channels": 32,
                        "max_groups": 16,
                        "bias": True,
                        "downsample_scales": [2, 2, 1],
                        "nonlinear_activation": "LeakyReLU",
                        "nonlinear_activation_params": {"negative_slope": 0.1},
                    },
                },
            },
            {},
        ),
        (
            {},
            {
                "discriminator_type": "hifigan_multi_period_discriminator",
                "discriminator_params": {
                    "periods": [2, 3],
                    "discriminator_params": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "kernel_sizes": [5, 3],
                        "channels": 16,
                        "downsample_scales": [3, 3, 1],
                        "max_downsample_channels": 32,
                        "bias": True,
                        "nonlinear_activation": "LeakyReLU",
                        "nonlinear_activation_params": {"negative_slope": 0.1},
                        "use_weight_norm": True,
                        "use_spectral_norm": False,
                    },
                },
            },
            {},
        ),
        (
            {},
            {
                "discriminator_type": "hifigan_period_discriminator",
                "discriminator_params": {
                    "period": 2,
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [5, 3],
                    "channels": 16,
                    "downsample_scales": [3, 3, 1],
                    "max_downsample_channels": 32,
                    "bias": True,
                    "nonlinear_activation": "LeakyReLU",
                    "nonlinear_activation_params": {"negative_slope": 0.1},
                    "use_weight_norm": True,
                    "use_spectral_norm": False,
                },
            },
            {},
        ),
        (
            {},
            {
                "discriminator_type": "hifigan_scale_discriminator",
                "discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [15, 41, 5, 3],
                    "channels": 16,
                    "max_downsample_channels": 32,
                    "max_groups": 16,
                    "bias": True,
                    "downsample_scales": [2, 2, 1],
                    "nonlinear_activation": "LeakyReLU",
                    "nonlinear_activation_params": {"negative_slope": 0.1},
                },
            },
            {},
        ),
        (
            {},
            {},
            {
                "generator_adv_loss_params": {
                    "average_by_discriminators": True,
                    "loss_type": "mse",
                },
                "discriminator_adv_loss_params": {
                    "average_by_discriminators": True,
                    "loss_type": "mse",
                },
            },
        ),
        (
            {},
            {},
            {
                "generator_adv_loss_params": {
                    "average_by_discriminators": False,
                    "loss_type": "hinge",
                },
                "discriminator_adv_loss_params": {
                    "average_by_discriminators": False,
                    "loss_type": "hinge",
                },
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "spks, spk_embed_dim, langs", [(10, -1, -1), (-1, 5, -1), (-1, -1, 3), (4, 5, 3)]
)
def test_multi_speaker_jets_is_trainable_and_decodable_on_gpu(
    gen_dict, dis_dict, loss_dict, spks, spk_embed_dim, langs
):
    idim = 10
    odim = 5
    gen_args = make_jets_generator_args(**gen_dict)
    gen_args["generator_params"]["spks"] = spks
    gen_args["generator_params"]["langs"] = langs
    gen_args["generator_params"]["spk_embed_dim"] = spk_embed_dim
    dis_args = make_jets_discriminator_args(**dis_dict)
    loss_args = make_jets_loss_args(**loss_dict)
    model = JETS(
        idim=idim,
        odim=odim,
        **gen_args,
        **dis_args,
        **loss_args,
    )
    model.train()
    upsample_factor = model.generator.upsample_factor
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, 16, odim),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        speech=torch.randn(2, 16 * upsample_factor),
        speech_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
        pitch=torch.randn(2, 16, 1),
        pitch_lengths=torch.tensor([16, 13], dtype=torch.long),
        energy=torch.randn(2, 16, 1),
        energy_lengths=torch.tensor([16, 13], dtype=torch.long),
    )
    if spks > 0:
        inputs["sids"] = torch.randint(0, spks, (2, 1))
    if langs > 0:
        inputs["lids"] = torch.randint(0, langs, (2, 1))
    if spk_embed_dim > 0:
        inputs["spembs"] = torch.randn(2, spk_embed_dim)
    device = torch.device("cuda")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_loss = model(forward_generator=True, **inputs)["loss"]
    gen_loss.backward()
    dis_loss = model(forward_generator=False, **inputs)["loss"]
    dis_loss.backward()

    with torch.no_grad():
        model.eval()

        # check inference
        inputs = dict(
            text=torch.randint(
                0,
                idim,
                (5,),
            ),
        )
        if spks > 0:
            inputs["sids"] = torch.randint(0, spks, (1,))
        if langs > 0:
            inputs["lids"] = torch.randint(0, langs, (1,))
        if spk_embed_dim > 0:
            inputs["spembs"] = torch.randn(spk_embed_dim)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.inference(**inputs)

        # check inference with teacher forcing
        inputs = dict(
            text=torch.randint(
                0,
                idim,
                (5,),
            ),
            feats=torch.randn(16, odim),
            pitch=torch.randn(16, 1),
            energy=torch.randn(16, 1),
        )
        if spks > 0:
            inputs["sids"] = torch.randint(0, spks, (1,))
        if langs > 0:
            inputs["lids"] = torch.randint(0, langs, (1,))
        if spk_embed_dim > 0:
            inputs["spembs"] = torch.randn(spk_embed_dim)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_dict = model.inference(**inputs, use_teacher_forcing=True)
        assert output_dict["wav"].size(0) == inputs["feats"].size(0) * upsample_factor
