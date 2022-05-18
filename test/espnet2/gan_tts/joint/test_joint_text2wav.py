# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test VITS related modules."""

from packaging.version import parse as V

import pytest
import torch

from espnet2.gan_tts.joint import JointText2Wav


def make_text2mel_args(**kwargs):
    defaults = dict(
        text2mel_type="fastspeech2",
        text2mel_params={
            "adim": 4,
            "aheads": 2,
            "elayers": 2,
            "eunits": 4,
            "dlayers": 2,
            "dunits": 3,
            "postnet_layers": 2,
            "postnet_chans": 4,
            "postnet_filts": 5,
            "postnet_dropout_rate": 0.5,
            "positionwise_layer_type": "conv1d",
            "positionwise_conv_kernel_size": 1,
            "use_scaled_pos_enc": True,
            "use_batch_norm": True,
            "encoder_normalize_before": True,
            "decoder_normalize_before": True,
            "encoder_concat_after": False,
            "decoder_concat_after": False,
            "reduction_factor": 1,
            "encoder_type": "conformer",
            "decoder_type": "conformer",
            "transformer_enc_dropout_rate": 0.1,
            "transformer_enc_positional_dropout_rate": 0.1,
            "transformer_enc_attn_dropout_rate": 0.1,
            "transformer_dec_dropout_rate": 0.1,
            "transformer_dec_positional_dropout_rate": 0.1,
            "transformer_dec_attn_dropout_rate": 0.1,
            "conformer_rel_pos_type": "latest",
            "conformer_pos_enc_layer_type": "rel_pos",
            "conformer_self_attn_layer_type": "rel_selfattn",
            "conformer_activation_type": "swish",
            "use_macaron_style_in_conformer": True,
            "use_cnn_in_conformer": True,
            "zero_triu": False,
            "conformer_enc_kernel_size": 7,
            "conformer_dec_kernel_size": 31,
            "duration_predictor_layers": 2,
            "duration_predictor_chans": 4,
            "duration_predictor_kernel_size": 3,
            "duration_predictor_dropout_rate": 0.1,
            "energy_predictor_layers": 2,
            "energy_predictor_chans": 4,
            "energy_predictor_kernel_size": 3,
            "energy_predictor_dropout": 0.5,
            "energy_embed_kernel_size": 1,
            "energy_embed_dropout": 0.5,
            "stop_gradient_from_energy_predictor": False,
            "pitch_predictor_layers": 2,
            "pitch_predictor_chans": 4,
            "pitch_predictor_kernel_size": 5,
            "pitch_predictor_dropout": 0.5,
            "pitch_embed_kernel_size": 1,
            "pitch_embed_dropout": 0.5,
            "stop_gradient_from_pitch_predictor": True,
            "spks": -1,
            "langs": -1,
            "spk_embed_dim": None,
            "spk_embed_integration_type": "add",
            "use_gst": False,
            "init_type": "xavier_uniform",
            "init_enc_alpha": 1.0,
            "init_dec_alpha": 1.0,
            "use_masking": False,
            "use_weighted_masking": False,
        },
    )
    defaults.update(kwargs)
    return defaults


def make_vocoder_args(**kwargs):
    defaults = dict(
        vocoder_type="hifigan_generator",
        vocoder_params={
            "out_channels": 1,
            "channels": 32,
            "global_channels": -1,
            "kernel_size": 7,
            "upsample_scales": [2, 2],
            "upsample_kernel_sizes": [4, 4],
            "resblock_kernel_sizes": [3, 7],
            "resblock_dilations": [[1, 3], [1, 3]],
            "use_additional_convs": True,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
        },
    )
    defaults.update(kwargs)
    return defaults


def make_discriminator_args(**kwargs):
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
            "follow_official_norm": True,
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


def make_loss_args(**kwargs):
    defaults = dict(
        lambda_text2mel=1.0,
        lambda_adv=1.0,
        lambda_feat_match=2.0,
        lambda_mel=1.0,
        generator_adv_loss_params={
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        discriminator_adv_loss_params={
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        use_feat_match_loss=True,
        feat_match_loss_params={
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        use_mel_loss=True,
        mel_loss_params={
            "fs": 22050,
            "n_fft": 16,
            "hop_length": 4,
            "win_length": None,
            "window": "hann",
            "n_mels": 4,
            "fmin": 0,
            "fmax": None,
            "log_base": None,
        },
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.skipif(
    V(torch.__version__) < V("1.4"),
    reason="Pytorch >= 1.4 is required.",
)
@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "t2m_dict, voc_dict, dis_dict, loss_dict",
    [
        ({}, {}, {}, {}),
        (
            {
                "text2mel_type": "tacotron2",
                "text2mel_params": {
                    "embed_dim": 8,
                    "elayers": 1,
                    "eunits": 8,
                    "econv_layers": 2,
                    "econv_chans": 8,
                    "econv_filts": 5,
                    "atype": "location",
                    "adim": 4,
                    "aconv_chans": 4,
                    "aconv_filts": 3,
                    "cumulate_att_w": True,
                    "dlayers": 2,
                    "dunits": 8,
                    "prenet_layers": 2,
                    "prenet_units": 4,
                    "postnet_layers": 2,
                    "postnet_chans": 4,
                    "postnet_filts": 3,
                    "output_activation": None,
                    "use_batch_norm": True,
                    "use_concate": True,
                    "use_residual": False,
                    "reduction_factor": 1,
                    "spk_embed_dim": None,
                    "spk_embed_integration_type": "concat",
                    "use_gst": False,
                    "dropout_rate": 0.5,
                    "zoneout_rate": 0.1,
                    "use_masking": True,
                    "use_weighted_masking": False,
                    "bce_pos_weight": 5.0,
                    "loss_type": "L1+L2",
                    "use_guided_attn_loss": True,
                    "guided_attn_loss_sigma": 0.4,
                    "guided_attn_loss_lambda": 1.0,
                },
            },
            {},
            {},
            {},
        ),
        (
            {
                "text2mel_type": "transformer",
                "text2mel_params": {
                    "embed_dim": 4,
                    "eprenet_conv_layers": 2,
                    "eprenet_conv_chans": 4,
                    "eprenet_conv_filts": 3,
                    "dprenet_layers": 2,
                    "dprenet_units": 7,
                    "elayers": 2,
                    "eunits": 4,
                    "adim": 4,
                    "aheads": 2,
                    "dlayers": 2,
                    "dunits": 3,
                    "postnet_layers": 1,
                    "postnet_chans": 2,
                    "postnet_filts": 3,
                    "positionwise_layer_type": "conv1d",
                    "positionwise_conv_kernel_size": 1,
                    "reduction_factor": 1,
                    "spk_embed_dim": None,
                    "use_gst": False,
                },
            },
            {},
            {},
            {},
        ),
        (
            {
                "text2mel_type": "fastspeech",
                "text2mel_params": {
                    # network structure related
                    "adim": 4,
                    "aheads": 2,
                    "elayers": 2,
                    "eunits": 3,
                    "dlayers": 2,
                    "dunits": 3,
                    "postnet_layers": 2,
                    "postnet_chans": 3,
                    "postnet_filts": 5,
                    "positionwise_layer_type": "conv1d",
                    "positionwise_conv_kernel_size": 1,
                    "use_scaled_pos_enc": True,
                    "use_batch_norm": True,
                    "encoder_normalize_before": True,
                    "decoder_normalize_before": True,
                    "encoder_concat_after": False,
                    "decoder_concat_after": False,
                    "duration_predictor_layers": 2,
                    "duration_predictor_chans": 3,
                    "duration_predictor_kernel_size": 3,
                    "reduction_factor": 1,
                    "encoder_type": "transformer",
                    "decoder_type": "transformer",
                    "spk_embed_dim": None,
                    "use_gst": False,
                },
            },
            {},
            {},
            {},
        ),
        (
            {},
            {
                "vocoder_type": "parallel_wavegan_generator",
                "vocoder_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_size": 3,
                    "layers": 6,
                    "stacks": 3,
                    "residual_channels": 8,
                    "gate_channels": 16,
                    "skip_channels": 8,
                    "aux_channels": 5,
                    "aux_context_window": 0,
                    "upsample_net": "ConvInUpsampleNetwork",
                    "upsample_params": {"upsample_scales": [4, 4]},
                },
            },
            {},
            {},
        ),
        (
            {},
            {},
            {
                "discriminator_type": "parallel_wavegan_discriminator",
                "discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_size": 3,
                    "layers": 5,
                    "conv_channels": 16,
                },
            },
            {},
        ),
        (
            {},
            {
                "vocoder_type": "melgan_generator",
                "vocoder_params": {
                    "in_channels": 5,
                    "out_channels": 1,
                    "kernel_size": 7,
                    "channels": 32,
                    "bias": True,
                    "upsample_scales": [4, 2],
                    "stack_kernel_size": 3,
                    "stacks": 1,
                    "pad": "ReplicationPad1d",
                },
            },
            {},
            {},
        ),
        (
            {},
            {},
            {
                "discriminator_type": "melgan_multi_scale_discriminator",
                "discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "scales": 2,
                    "kernel_sizes": [5, 3],
                    "channels": 16,
                    "max_downsample_channels": 32,
                    "bias": True,
                    "downsample_scales": [2, 2],
                },
            },
            {},
        ),
        (
            {},
            {
                "vocoder_type": "style_melgan_generator",
                "vocoder_params": {
                    "in_channels": 32,
                    "aux_channels": 5,
                    "channels": 16,
                    "out_channels": 1,
                    "kernel_size": 9,
                    "dilation": 2,
                    "bias": True,
                    "noise_upsample_scales": [2, 2],
                    "noise_upsample_activation": "LeakyReLU",
                    "noise_upsample_activation_params": {"negative_slope": 0.2},
                    "upsample_scales": [4, 4],
                },
            },
            {},
            {},
        ),
        (
            {},
            {},
            {
                "discriminator_type": "style_melgan_discriminator",
                "discriminator_params": {
                    "repeats": 2,
                    "window_sizes": [4, 8],
                    "pqmf_params": [
                        [1, None, None, None],
                        [2, 62, 0.26700, 9.0],
                    ],
                    "discriminator_params": {
                        "out_channels": 1,
                        "kernel_sizes": [5, 3],
                        "channels": 16,
                        "max_downsample_channels": 32,
                        "bias": True,
                        "downsample_scales": [2, 2],
                        "nonlinear_activation": "LeakyReLU",
                        "nonlinear_activation_params": {"negative_slope": 0.2},
                        "pad": "ReplicationPad1d",
                        "pad_params": {},
                    },
                    "use_weight_norm": True,
                },
            },
            {},
        ),
        (
            {},
            {
                "vocoder_params": {
                    "out_channels": 4,
                    "channels": 32,
                    "global_channels": -1,
                    "kernel_size": 7,
                    "upsample_scales": [4, 2],
                    "upsample_kernel_sizes": [8, 4],
                    "resblock_kernel_sizes": [3, 7],
                    "resblock_dilations": [[1, 3], [1, 3]],
                },
                "use_pqmf": True,
            },
            {},
            {},
        ),
        (
            {},
            {
                "vocoder_type": "melgan_generator",
                "vocoder_params": {
                    "in_channels": 5,
                    "out_channels": 4,
                    "kernel_size": 7,
                    "channels": 32,
                    "bias": True,
                    "upsample_scales": [4, 2],
                    "stack_kernel_size": 3,
                    "stacks": 1,
                    "pad": "ReplicationPad1d",
                },
                "use_pqmf": True,
            },
            {},
            {},
        ),
    ],
)
def test_joint_model_is_trainable_and_decodable(
    t2m_dict, voc_dict, dis_dict, loss_dict
):
    idim = 10
    odim = 5
    t2m_args = make_text2mel_args(**t2m_dict)
    voc_args = make_vocoder_args(**voc_dict)
    dis_args = make_discriminator_args(**dis_dict)
    loss_args = make_loss_args(**loss_dict)
    model = JointText2Wav(
        idim=idim,
        odim=odim,
        segment_size=4,
        **t2m_args,
        **voc_args,
        **dis_args,
        **loss_args,
    )
    model.train()
    upsample_factor = model.generator["vocoder"].upsample_factor
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, 16, odim),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        speech=torch.randn(2, 16 * upsample_factor),
        speech_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
    )
    if t2m_args["text2mel_type"] in ["fastspeech", "fastspeech2"]:
        inputs.update(
            durations=torch.tensor(
                [
                    # +1 element for <eos>
                    [2, 2, 2, 2, 2, 2, 2, 2, 0],
                    [3, 3, 3, 3, 1, 0, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
            # +1 element for <eos>
            durations_lengths=torch.tensor([8 + 1, 5 + 1], dtype=torch.long),
        )
    if t2m_args["text2mel_type"] in ["fastspeech2"]:
        inputs.update(
            pitch=torch.randn(2, 9, 1),
            pitch_lengths=torch.tensor([9, 7], dtype=torch.long),
            energy=torch.randn(2, 9, 1),
            energy_lengths=torch.tensor([9, 7], dtype=torch.long),
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
                (10,),
            )
        )
        output_dict = model.inference(**inputs)
        assert len(output_dict["wav"]) == len(output_dict["feat_gen"]) * upsample_factor
