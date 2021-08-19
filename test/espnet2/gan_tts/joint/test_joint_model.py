# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test VITS related modules."""

from distutils.version import LooseVersion

import pytest
import torch

from espnet2.gan_tts.joint.joint_model import JointText2Wav


def make_text2mel_args(**kwargs):
    defaults = dict(
        text2mel_type="tacotron2",
        text2mel_params={
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
            "gst_tokens": 10,
            "gst_heads": 4,
            "gst_conv_layers": 6,
            "gst_conv_chans_list": [32, 32, 64, 64, 128, 128],
            "gst_conv_kernel_size": 3,
            "gst_conv_stride": 2,
            "gst_gru_layers": 1,
            "gst_gru_units": 128,
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
            "upsample_kernal_sizes": [4, 4],
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
            "scales": 2,
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
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.4"),
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
    upsample_factor = model.vocoder.upsample_factor
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, 16, odim),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        speech=torch.randn(2, 16 * upsample_factor),
        speech_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
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
        output_dict = model.inference(**inputs)
        assert len(output_dict["wav"]) == len(output_dict["feat_gen"]) * upsample_factor
