# Copyright 2021 Tomoki Hayashi
# Copyright 2023 Yuning Wu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test joint score2wav related modules."""

import pytest
import torch
from packaging.version import parse as V

from espnet2.gan_svs.joint import JointScore2Wav


def make_score2mel_args(**kwargs):
    defaults = dict(
        score2mel_type="xiaoice",
        score2mel_params={
            "midi_dim": 129,
            "duration_dim": 10,
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
            "spks": -1,
            "langs": -1,
            "spk_embed_dim": None,
            "spk_embed_integration_type": "add",
            "init_type": "xavier_uniform",
            "init_enc_alpha": 1.0,
            "init_dec_alpha": 1.0,
            "use_masking": False,
            "use_weighted_masking": False,
            "loss_function": "XiaoiceSing2",
            "loss_type": "L1",
            "lambda_mel": 1,
            "lambda_dur": 0.1,
            "lambda_pitch": 0.01,
            "lambda_vuv": 0.01,
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
        lambda_score2mel=1.0,
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
    "s2m_dict, voc_dict, dis_dict, loss_dict",
    [
        ({}, {}, {}, {}),
        (
            {
                "score2mel_type": "naive_rnn_dp",
                "score2mel_params": {
                    "midi_dim": 129,
                    "duration_dim": 10,
                    "embed_dim": 8,
                    "eprenet_conv_layers": 2,
                    "eprenet_conv_chans": 8,
                    "eprenet_conv_filts": 5,
                    "elayers": 1,
                    "eunits": 8,
                    "ebidirectional": True,
                    "midi_embed_integration_type": "add",
                    "dlayers": 2,
                    "dunits": 8,
                    "dbidirectional": True,
                    "postnet_layers": 2,
                    "postnet_chans": 4,
                    "postnet_filts": 3,
                    "use_batch_norm": True,
                    "duration_predictor_layers": 2,
                    "duration_predictor_chans": 4,
                    "duration_predictor_kernel_size": 3,
                    "duration_predictor_dropout_rate": 0.1,
                    "reduction_factor": 1,
                    "spks": -1,
                    "langs": -1,
                    "spk_embed_dim": None,
                    "spk_embed_integration_type": "concat",
                    "eprenet_dropout_rate": 0.5,
                    "edropout_rate": 0.1,
                    "ddropout_rate": 0.1,
                    "postnet_dropout_rate": 0.5,
                    "init_type": "xavier_uniform",
                    "use_masking": False,
                    "use_weighted_masking": False,
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
    s2m_dict, voc_dict, dis_dict, loss_dict
):
    idim = 10
    odim = 5
    s2m_args = make_score2mel_args(**s2m_dict)
    voc_args = make_vocoder_args(**voc_dict)
    dis_args = make_discriminator_args(**dis_dict)
    loss_args = make_loss_args(**loss_dict)
    model = JointScore2Wav(
        idim=idim,
        odim=odim,
        segment_size=4,
        **s2m_args,
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
        feats_lengths=torch.tensor([16, 10], dtype=torch.long),
        singing=torch.randn(2, 16 * upsample_factor),
        singing_lengths=torch.tensor([16, 10] * upsample_factor, dtype=torch.long),
        label={
            "lab": torch.randint(0, idim, (2, 8)),
            "score": torch.randint(0, idim, (2, 8)),
        },
        label_lengths={
            "lab": torch.tensor([8, 5], dtype=torch.long),
            "score": torch.tensor([8, 5], dtype=torch.long),
        },
        melody={
            "lab": torch.randint(0, 127, (2, 8)),
            "score": torch.randint(0, 127, (2, 8)),
        },
        duration={
            "lab": torch.tensor(
                [[1, 2, 2, 3, 1, 3, 2, 2], [2, 2, 1, 4, 1, 2, 1, 3]], dtype=torch.int64
            ),
            "score_phn": torch.tensor(
                [[1, 2, 2, 3, 1, 3, 2, 1], [2, 2, 1, 4, 1, 2, 1, 3]], dtype=torch.int64
            ),
            "score_syb": torch.tensor(
                [[3, 3, 5, 5, 4, 4, 3, 3], [4, 4, 5, 5, 3, 3, 4, 4]], dtype=torch.int64
            ),
        },
        slur=torch.randint(0, 2, (2, 8)),
        pitch=torch.randn(2, 16, 1),
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
                (
                    1,
                    5,
                ),
            ),
            label={
                "lab": torch.randint(
                    0,
                    idim,
                    (
                        1,
                        5,
                    ),
                ),
                "score": torch.randint(
                    0,
                    idim,
                    (
                        1,
                        5,
                    ),
                ),
            },
            melody={
                "lab": torch.randint(
                    0,
                    127,
                    (
                        1,
                        5,
                    ),
                ),
                "score": torch.randint(
                    0,
                    127,
                    (
                        1,
                        5,
                    ),
                ),
            },
            duration={
                "lab": torch.tensor([[1, 2, 2, 3, 3]], dtype=torch.int64),
                "score_phn": torch.tensor([[1, 2, 2, 3, 4]], dtype=torch.int64),
                "score_syb": torch.tensor([[3, 3, 5, 5, 4]], dtype=torch.int64),
            },
            slur=torch.randint(0, 2, (1, 5)),
            pitch=torch.randn(11, 1),
        )
        output_dict = model.inference(**inputs)
        assert len(output_dict["wav"]) == len(output_dict["feat_gen"]) * upsample_factor
