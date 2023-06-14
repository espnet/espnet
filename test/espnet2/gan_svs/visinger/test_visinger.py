# Copyright 2021 Tomoki Hayashi
# Copyright 2023 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test VISinger related modules."""

import pytest
import torch

from espnet2.gan_svs.vits import VITS


def get_test_data():
    test_data = [
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
        # visinger 2 discriminator (mpd+msd+mfd)
        (
            {},
            {
                "discriminator_type": "visinger2",
                "discriminator_params": {
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
                    "multi_freq_disc_params": {
                        "hop_length_factors": [1, 2, 4],
                        "hidden_channels": [2, 2, 4],
                        "domain": "double",
                        "mel_scale": True,
                        "divisors": [1, 1, 1, 1, 1, 1, 1],
                        "strides": [1, 2, 1, 2, 1, 2, 1],
                    },
                },
            },
            {},
        ),
        # avocodo discriminator
        (
            {
                "generator_type": "visinger",
                "vocoder_generator_type": "hifigan",
                "generator_params": {
                    "vocabs": 10,
                    "aux_channels": 5,
                    "hidden_channels": 4,
                    "spks": -1,
                    "langs": -1,
                    "spk_embed_dim": -1,
                    "global_channels": -1,
                    "segment_size": 4,
                    "text_encoder_attention_heads": 2,
                    "text_encoder_ffn_expand": 2,
                    "text_encoder_blocks": 2,
                    "text_encoder_positionwise_layer_type": "conv1d",
                    "text_encoder_positionwise_conv_kernel_size": 1,
                    "text_encoder_positional_encoding_layer_type": "rel_pos",
                    "text_encoder_self_attention_layer_type": "rel_selfattn",
                    "text_encoder_activation_type": "swish",
                    "text_encoder_normalize_before": True,
                    "text_encoder_dropout_rate": 0.1,
                    "text_encoder_positional_dropout_rate": 0.0,
                    "text_encoder_attention_dropout_rate": 0.0,
                    "text_encoder_conformer_kernel_size": 7,
                    "use_macaron_style_in_text_encoder": True,
                    "use_conformer_conv_in_text_encoder": True,
                    "decoder_kernel_size": 7,
                    "decoder_channels": 16,
                    "decoder_upsample_scales": (2, 2, 4, 16),
                    "decoder_upsample_kernel_sizes": (4, 4, 8, 32),
                    "decoder_resblock_kernel_sizes": (3, 5),
                    "decoder_resblock_dilations": [(1, 3), (1, 3)],
                    "use_weight_norm_in_decoder": True,
                    "posterior_encoder_kernel_size": 5,
                    "posterior_encoder_layers": 2,
                    "posterior_encoder_stacks": 1,
                    "posterior_encoder_base_dilation": 1,
                    "posterior_encoder_dropout_rate": 0.0,
                    "use_weight_norm_in_posterior_encoder": True,
                    "flow_flows": 2,
                    "flow_kernel_size": 5,
                    "flow_base_dilation": 1,
                    "flow_layers": 2,
                    "flow_dropout_rate": 0.0,
                    "use_weight_norm_in_flow": True,
                    "use_only_mean_in_flow": True,
                    "fs": 22050,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_fft": 1024,
                    "use_phoneme_predictor": False,
                    # avocodo
                    "projection_filters": [0, 1, 1, 1],
                    "projection_kernels": [0, 5, 7, 11],
                },
            },
            {
                "discriminator_type": "avocodo",
                "discriminator_params": {
                    "combd": {
                        "combd_h_u": [
                            [16, 64],
                            [16, 64],
                        ],
                        "combd_d_k": [
                            [3, 3],
                            [3, 3],
                        ],
                        "combd_d_s": [
                            [1, 1],
                            [1, 1],
                        ],
                        "combd_d_d": [
                            [1, 1],
                            [1, 1],
                        ],
                        "combd_d_g": [
                            [1, 4],
                            [1, 4],
                        ],
                        "combd_d_p": [
                            [3, 5],
                            [5, 10],
                        ],
                        "combd_op_f": [1, 1],
                        "combd_op_k": [3, 3],
                        # combd_op_k: [1, 3] # change for uhifigan 1 avocodo
                        "combd_op_g": [1, 1],
                        "use_spectral_norm": False,
                    },
                    "sbd": {
                        "sbd_filters": [
                            [64, 128],
                            [32, 64],
                        ],
                        "sbd_strides": [
                            [1, 1],
                            [1, 1],
                        ],
                        "sbd_kernel_sizes": [
                            [[5, 5, 5], [5, 5, 5]],
                            [[3, 3, 3], [3, 3, 3]],
                        ],
                        "sbd_dilations": [
                            [[5, 7, 11], [5, 7, 11]],
                            [[1, 2, 3], [1, 2, 3]],
                        ],
                        "sbd_band_ranges": [[0, 6], [0, 64]],
                        "sbd_transpose": [False, True],
                        "pqmf_config": {
                            "sbd": [16, 256, 0.03, 10.0],
                            "fsbd": [64, 256, 0.1, 9.0],
                        },
                        "segment_size": 1024,  # 4 * hop_size
                        "use_spectral_norm": False,
                    },
                    "pqmf_config": {
                        "lv1": [16, 256, 0.25, 10.0],
                        "lv2": [64, 192, 0.13, 10.0],
                    },
                },
            },
            {},
        ),
        # visinger 2 vocoder (ddsp)
        (
            {
                "generator_type": "visinger",
                "vocoder_generator_type": "visinger2",
                "generator_params": {
                    "vocabs": 10,
                    "aux_channels": 5,
                    "hidden_channels": 4,
                    "spks": -1,
                    "langs": -1,
                    "spk_embed_dim": -1,
                    "global_channels": -1,
                    "segment_size": 4,
                    "text_encoder_attention_heads": 2,
                    "text_encoder_ffn_expand": 2,
                    "text_encoder_blocks": 2,
                    "text_encoder_positionwise_layer_type": "conv1d",
                    "text_encoder_positionwise_conv_kernel_size": 1,
                    "text_encoder_positional_encoding_layer_type": "rel_pos",
                    "text_encoder_self_attention_layer_type": "rel_selfattn",
                    "text_encoder_activation_type": "swish",
                    "text_encoder_normalize_before": True,
                    "text_encoder_dropout_rate": 0.1,
                    "text_encoder_positional_dropout_rate": 0.0,
                    "text_encoder_attention_dropout_rate": 0.0,
                    "text_encoder_conformer_kernel_size": 7,
                    "use_macaron_style_in_text_encoder": True,
                    "use_conformer_conv_in_text_encoder": True,
                    "decoder_kernel_size": 7,
                    "decoder_channels": 16,
                    "decoder_upsample_scales": (16, 16),
                    "decoder_upsample_kernel_sizes": (32, 32),
                    "decoder_resblock_kernel_sizes": (3, 5),
                    "decoder_resblock_dilations": [(1, 3), (1, 3)],
                    "use_weight_norm_in_decoder": True,
                    "posterior_encoder_kernel_size": 5,
                    "posterior_encoder_layers": 2,
                    "posterior_encoder_stacks": 1,
                    "posterior_encoder_base_dilation": 1,
                    "posterior_encoder_dropout_rate": 0.0,
                    "use_weight_norm_in_posterior_encoder": True,
                    "flow_flows": 2,
                    "flow_kernel_size": 5,
                    "flow_base_dilation": 1,
                    "flow_layers": 2,
                    "flow_dropout_rate": 0.0,
                    "use_weight_norm_in_flow": True,
                    "use_only_mean_in_flow": True,
                    "fs": 22050,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_fft": 1024,
                    "use_phoneme_predictor": False,
                },
            },
            {
                "discriminator_type": "hifigan_multi_scale_multi_period_discriminator",
                "discriminator_params": {
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
            },
            {},
        ),
        # uhifigan vocoder
        (
            {
                "generator_type": "visinger",
                "vocoder_generator_type": "uhifigan",
                "generator_params": {
                    "vocabs": 10,
                    "aux_channels": 5,
                    "hidden_channels": 4,
                    "spks": -1,
                    "langs": -1,
                    "spk_embed_dim": -1,
                    "global_channels": -1,
                    "segment_size": 4,
                    "text_encoder_attention_heads": 2,
                    "text_encoder_ffn_expand": 2,
                    "text_encoder_blocks": 2,
                    "text_encoder_positionwise_layer_type": "conv1d",
                    "text_encoder_positionwise_conv_kernel_size": 1,
                    "text_encoder_positional_encoding_layer_type": "rel_pos",
                    "text_encoder_self_attention_layer_type": "rel_selfattn",
                    "text_encoder_activation_type": "swish",
                    "text_encoder_normalize_before": True,
                    "text_encoder_dropout_rate": 0.1,
                    "text_encoder_positional_dropout_rate": 0.0,
                    "text_encoder_attention_dropout_rate": 0.0,
                    "text_encoder_conformer_kernel_size": 7,
                    "use_macaron_style_in_text_encoder": True,
                    "use_conformer_conv_in_text_encoder": True,
                    "decoder_kernel_size": 7,
                    "decoder_channels": 16,
                    "decoder_downsample_scales": (16, 16),
                    "decoder_downsample_kernel_sizes": (32, 32),
                    "decoder_upsample_scales": (16, 16),
                    "decoder_upsample_kernel_sizes": (32, 32),
                    "decoder_resblock_kernel_sizes": (3, 5),
                    "decoder_resblock_dilations": [(1, 3), (1, 3)],
                    "use_weight_norm_in_decoder": True,
                    "posterior_encoder_kernel_size": 5,
                    "posterior_encoder_layers": 2,
                    "posterior_encoder_stacks": 1,
                    "posterior_encoder_base_dilation": 1,
                    "posterior_encoder_dropout_rate": 0.0,
                    "use_weight_norm_in_posterior_encoder": True,
                    "flow_flows": 2,
                    "flow_kernel_size": 5,
                    "flow_base_dilation": 1,
                    "flow_layers": 2,
                    "flow_dropout_rate": 0.0,
                    "use_weight_norm_in_flow": True,
                    "use_only_mean_in_flow": True,
                    "fs": 22050,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_fft": 1024,
                    "use_phoneme_predictor": False,
                },
            },
            {
                "discriminator_type": "hifigan_multi_scale_multi_period_discriminator",
                "discriminator_params": {
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
            },
            {},
        ),
        # visinger 2 generator
        (
            {
                "generator_type": "visinger2",
                "vocoder_generator_type": "hifigan",
                "generator_params": {
                    "vocabs": 10,
                    "aux_channels": 5,
                    "hidden_channels": 4,
                    "spks": -1,
                    "langs": -1,
                    "spk_embed_dim": -1,
                    "global_channels": -1,
                    "segment_size": 4,
                    "text_encoder_attention_heads": 2,
                    "text_encoder_ffn_expand": 2,
                    "text_encoder_blocks": 2,
                    "text_encoder_positionwise_layer_type": "conv1d",
                    "text_encoder_positionwise_conv_kernel_size": 1,
                    "text_encoder_positional_encoding_layer_type": "rel_pos",
                    "text_encoder_self_attention_layer_type": "rel_selfattn",
                    "text_encoder_activation_type": "swish",
                    "text_encoder_normalize_before": True,
                    "text_encoder_dropout_rate": 0.1,
                    "text_encoder_positional_dropout_rate": 0.0,
                    "text_encoder_attention_dropout_rate": 0.0,
                    "text_encoder_conformer_kernel_size": 7,
                    "use_macaron_style_in_text_encoder": True,
                    "use_conformer_conv_in_text_encoder": True,
                    "decoder_kernel_size": 7,
                    "decoder_channels": 16,
                    "decoder_downsample_scales": (16, 16),
                    "decoder_downsample_kernel_sizes": (32, 32),
                    "decoder_upsample_scales": (16, 16),
                    "decoder_upsample_kernel_sizes": (32, 32),
                    "decoder_resblock_kernel_sizes": (3, 5),
                    "decoder_resblock_dilations": [(1, 3), (1, 3)],
                    "use_weight_norm_in_decoder": True,
                    "posterior_encoder_kernel_size": 5,
                    "posterior_encoder_layers": 2,
                    "posterior_encoder_stacks": 1,
                    "posterior_encoder_base_dilation": 1,
                    "posterior_encoder_dropout_rate": 0.0,
                    "use_weight_norm_in_posterior_encoder": True,
                    "flow_flows": 2,
                    "flow_kernel_size": 5,
                    "flow_base_dilation": 1,
                    "flow_layers": 2,
                    "flow_dropout_rate": 0.0,
                    "use_weight_norm_in_flow": True,
                    "use_only_mean_in_flow": True,
                    "fs": 22050,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_fft": 1024,
                    "use_phoneme_predictor": False,
                },
            },
            {
                "discriminator_type": "hifigan_multi_scale_multi_period_discriminator",
                "discriminator_params": {
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
            },
            {},
        ),
        # without flow
        (
            {
                "generator_type": "visinger",
                "vocoder_generator_type": "hifigan",
                "generator_params": {
                    "vocabs": 10,
                    "aux_channels": 5,
                    "hidden_channels": 4,
                    "spks": -1,
                    "langs": -1,
                    "spk_embed_dim": -1,
                    "global_channels": -1,
                    "segment_size": 4,
                    "text_encoder_attention_heads": 2,
                    "text_encoder_ffn_expand": 2,
                    "text_encoder_blocks": 2,
                    "text_encoder_positionwise_layer_type": "conv1d",
                    "text_encoder_positionwise_conv_kernel_size": 1,
                    "text_encoder_positional_encoding_layer_type": "rel_pos",
                    "text_encoder_self_attention_layer_type": "rel_selfattn",
                    "text_encoder_activation_type": "swish",
                    "text_encoder_normalize_before": True,
                    "text_encoder_dropout_rate": 0.1,
                    "text_encoder_positional_dropout_rate": 0.0,
                    "text_encoder_attention_dropout_rate": 0.0,
                    "text_encoder_conformer_kernel_size": 7,
                    "use_macaron_style_in_text_encoder": True,
                    "use_conformer_conv_in_text_encoder": True,
                    "decoder_kernel_size": 7,
                    "decoder_channels": 16,
                    "decoder_downsample_scales": (16, 16),
                    "decoder_downsample_kernel_sizes": (32, 32),
                    "decoder_upsample_scales": (16, 16),
                    "decoder_upsample_kernel_sizes": (32, 32),
                    "decoder_resblock_kernel_sizes": (3, 5),
                    "decoder_resblock_dilations": [(1, 3), (1, 3)],
                    "use_weight_norm_in_decoder": True,
                    "posterior_encoder_kernel_size": 5,
                    "posterior_encoder_layers": 2,
                    "posterior_encoder_stacks": 1,
                    "posterior_encoder_base_dilation": 1,
                    "posterior_encoder_dropout_rate": 0.0,
                    "use_weight_norm_in_posterior_encoder": True,
                    "flow_flows": -1,
                    "flow_kernel_size": 5,
                    "flow_base_dilation": 1,
                    "flow_layers": 2,
                    "flow_dropout_rate": 0.0,
                    "use_weight_norm_in_flow": True,
                    "use_only_mean_in_flow": True,
                    "fs": 22050,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_fft": 1024,
                    "use_phoneme_predictor": False,
                },
            },
            {},
            {},
        ),
        # use phoneme predictor
        (
            {
                "generator_type": "visinger",
                "vocoder_generator_type": "hifigan",
                "generator_params": {
                    "vocabs": 10,
                    "aux_channels": 5,
                    "hidden_channels": 4,
                    "spks": -1,
                    "langs": -1,
                    "spk_embed_dim": -1,
                    "global_channels": -1,
                    "segment_size": 4,
                    "text_encoder_attention_heads": 2,
                    "text_encoder_ffn_expand": 2,
                    "text_encoder_blocks": 2,
                    "text_encoder_positionwise_layer_type": "conv1d",
                    "text_encoder_positionwise_conv_kernel_size": 1,
                    "text_encoder_positional_encoding_layer_type": "rel_pos",
                    "text_encoder_self_attention_layer_type": "rel_selfattn",
                    "text_encoder_activation_type": "swish",
                    "text_encoder_normalize_before": True,
                    "text_encoder_dropout_rate": 0.1,
                    "text_encoder_positional_dropout_rate": 0.0,
                    "text_encoder_attention_dropout_rate": 0.0,
                    "text_encoder_conformer_kernel_size": 7,
                    "use_macaron_style_in_text_encoder": True,
                    "use_conformer_conv_in_text_encoder": True,
                    "decoder_kernel_size": 7,
                    "decoder_channels": 16,
                    "decoder_downsample_scales": (16, 16),
                    "decoder_downsample_kernel_sizes": (32, 32),
                    "decoder_upsample_scales": (16, 16),
                    "decoder_upsample_kernel_sizes": (32, 32),
                    "decoder_resblock_kernel_sizes": (3, 5),
                    "decoder_resblock_dilations": [(1, 3), (1, 3)],
                    "use_weight_norm_in_decoder": True,
                    "posterior_encoder_kernel_size": 5,
                    "posterior_encoder_layers": 2,
                    "posterior_encoder_stacks": 1,
                    "posterior_encoder_base_dilation": 1,
                    "posterior_encoder_dropout_rate": 0.0,
                    "use_weight_norm_in_posterior_encoder": True,
                    "flow_flows": 2,
                    "flow_kernel_size": 5,
                    "flow_base_dilation": 1,
                    "flow_layers": 2,
                    "flow_dropout_rate": 0.0,
                    "use_weight_norm_in_flow": True,
                    "use_only_mean_in_flow": True,
                    "fs": 22050,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_fft": 1024,
                    "use_phoneme_predictor": True,
                },
            },
            {},
            {},
        ),
    ]
    return test_data


def make_vits_generator_args(**kwargs):
    defaults = dict(
        generator_type="visinger",
        vocoder_generator_type="hifigan",
        generator_params={
            "vocabs": 10,
            "aux_channels": 5,
            "hidden_channels": 4,
            "spks": -1,
            "langs": -1,
            "spk_embed_dim": -1,
            "global_channels": -1,
            "segment_size": 4,
            "text_encoder_attention_heads": 2,
            "text_encoder_ffn_expand": 2,
            "text_encoder_blocks": 2,
            "text_encoder_positionwise_layer_type": "conv1d",
            "text_encoder_positionwise_conv_kernel_size": 1,
            "text_encoder_positional_encoding_layer_type": "rel_pos",
            "text_encoder_self_attention_layer_type": "rel_selfattn",
            "text_encoder_activation_type": "swish",
            "text_encoder_normalize_before": True,
            "text_encoder_dropout_rate": 0.1,
            "text_encoder_positional_dropout_rate": 0.0,
            "text_encoder_attention_dropout_rate": 0.0,
            "text_encoder_conformer_kernel_size": 7,
            "use_macaron_style_in_text_encoder": True,
            "use_conformer_conv_in_text_encoder": True,
            "decoder_kernel_size": 7,
            "decoder_channels": 16,
            "decoder_upsample_scales": (16, 16),
            "decoder_upsample_kernel_sizes": (32, 32),
            "decoder_resblock_kernel_sizes": (3, 5),
            "decoder_resblock_dilations": [(1, 3), (1, 3)],
            "use_weight_norm_in_decoder": True,
            "posterior_encoder_kernel_size": 5,
            "posterior_encoder_layers": 2,
            "posterior_encoder_stacks": 1,
            "posterior_encoder_base_dilation": 1,
            "posterior_encoder_dropout_rate": 0.0,
            "use_weight_norm_in_posterior_encoder": True,
            "flow_flows": 2,
            "flow_kernel_size": 5,
            "flow_base_dilation": 1,
            "flow_layers": 2,
            "flow_dropout_rate": 0.0,
            "use_weight_norm_in_flow": True,
            "use_only_mean_in_flow": True,
            "generator_type": "visinger",
            "vocoder_generator_type": "hifigan",
            "fs": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "use_phoneme_predictor": False,
            # avocodo
            "projection_filters": [0, 1, 1, 1],
            "projection_kernels": [0, 5, 7, 11],
        },
    )
    defaults.update(kwargs)
    return defaults


def make_vits_discriminator_args(**kwargs):
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


def make_vits_loss_args(**kwargs):
    defaults = dict(
        lambda_adv=1.0,
        lambda_mel=45.0,
        lambda_feat_match=2.0,
        lambda_dur=1.0,
        lambda_kl=1.0,
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


@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "gen_dict, dis_dict, loss_dict",
    get_test_data(),
)
def test_vits_is_trainable_and_decodable(gen_dict, dis_dict, loss_dict):
    idim = 10
    odim = 5
    gen_args = make_vits_generator_args(**gen_dict)
    dis_args = make_vits_discriminator_args(**dis_dict)
    loss_args = make_vits_loss_args(**loss_dict)
    model = VITS(
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
        singing=torch.randn(2, 16 * upsample_factor),
        singing_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
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
            pitch=torch.randn(16, 1),
        )
        model.inference(**inputs)

        # check inference with teachder forcing
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
            pitch=torch.randn(16, 1),
            feats=torch.randn(16, odim),
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
    get_test_data(),
)
@pytest.mark.parametrize(
    "spks, spk_embed_dim, langs", [(10, -1, -1), (-1, 5, -1), (-1, -1, 3), (4, 5, 3)]
)
def test_multi_speaker_vits_is_trainable_and_decodable(
    gen_dict, dis_dict, loss_dict, spks, spk_embed_dim, langs
):
    idim = 10
    odim = 5
    global_channels = 8
    gen_args = make_vits_generator_args(**gen_dict)
    print("gen_args", gen_args)
    gen_args["generator_params"]["spks"] = spks
    gen_args["generator_params"]["langs"] = langs
    gen_args["generator_params"]["spk_embed_dim"] = spk_embed_dim
    gen_args["generator_params"]["global_channels"] = global_channels
    dis_args = make_vits_discriminator_args(**dis_dict)
    loss_args = make_vits_loss_args(**loss_dict)
    model = VITS(
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
        singing=torch.randn(2, 16 * upsample_factor),
        singing_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
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
            pitch=torch.randn(16, 1),
        )
        if spks > 0:
            inputs["sids"] = torch.randint(0, spks, (1,))
        if langs > 0:
            inputs["lids"] = torch.randint(0, langs, (1,))
        if spk_embed_dim > 0:
            inputs["spembs"] = torch.randn(spk_embed_dim)
        model.inference(**inputs)

        # check inference with teachder forcing
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
            pitch=torch.randn(16, 1),
            feats=torch.randn(16, odim),
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
    get_test_data(),
)
def test_vits_is_trainable_and_decodable_on_gpu(gen_dict, dis_dict, loss_dict):
    idim = 10
    odim = 5
    gen_args = make_vits_generator_args(**gen_dict)
    dis_args = make_vits_discriminator_args(**dis_dict)
    loss_args = make_vits_loss_args(**loss_dict)
    model = VITS(
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
        singing=torch.randn(2, 16 * upsample_factor),
        singing_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
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
    device = torch.device("cuda")
    model.to(device)
    inputs = {
        k: {k2: v2.to(device) for k2, v2 in v.items()}
        if isinstance(v, dict)
        else v.to(device)
        for k, v in inputs.items()
    }
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
            pitch=torch.randn(16, 1),
        )
        inputs = {
            k: {k2: v2.to(device) for k2, v2 in v.items()}
            if isinstance(v, dict)
            else v.to(device)
            for k, v in inputs.items()
        }
        model.inference(**inputs)

        # check inference with teachder forcing
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
            pitch=torch.randn(16, 1),
            feats=torch.randn(16, odim),
        )
        inputs = {
            k: {k2: v2.to(device) for k2, v2 in v.items()}
            if isinstance(v, dict)
            else v.to(device)
            for k, v in inputs.items()
        }
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
    get_test_data(),
)
@pytest.mark.parametrize(
    "spks, spk_embed_dim, langs", [(10, -1, -1), (-1, 5, -1), (-1, -1, 3), (4, 5, 3)]
)
def test_multi_speaker_vits_is_trainable_and_decodable_on_gpu(
    gen_dict, dis_dict, loss_dict, spks, spk_embed_dim, langs
):
    idim = 10
    odim = 5
    global_channels = 8
    gen_args = make_vits_generator_args(**gen_dict)
    gen_args["generator_params"]["spks"] = spks
    gen_args["generator_params"]["langs"] = langs
    gen_args["generator_params"]["spk_embed_dim"] = spk_embed_dim
    gen_args["generator_params"]["global_channels"] = global_channels
    dis_args = make_vits_discriminator_args(**dis_dict)
    loss_args = make_vits_loss_args(**loss_dict)
    model = VITS(
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
        singing=torch.randn(2, 16 * upsample_factor),
        singing_lengths=torch.tensor([16, 13] * upsample_factor, dtype=torch.long),
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
    if spks > 0:
        inputs["sids"] = torch.randint(0, spks, (2, 1))
    if langs > 0:
        inputs["lids"] = torch.randint(0, langs, (2, 1))
    if spk_embed_dim > 0:
        inputs["spembs"] = torch.randn(2, spk_embed_dim)
    device = torch.device("cuda")
    model.to(device)
    inputs = {
        k: {k2: v2.to(device) for k2, v2 in v.items()}
        if isinstance(v, dict)
        else v.to(device)
        for k, v in inputs.items()
    }
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
            pitch=torch.randn(16, 1),
        )
        if spks > 0:
            inputs["sids"] = torch.randint(0, spks, (1,))
        if langs > 0:
            inputs["lids"] = torch.randint(0, langs, (1,))
        if spk_embed_dim > 0:
            inputs["spembs"] = torch.randn(spk_embed_dim)
        inputs = {
            k: {k2: v2.to(device) for k2, v2 in v.items()}
            if isinstance(v, dict)
            else v.to(device)
            for k, v in inputs.items()
        }
        model.inference(**inputs)

        # check inference with teachder forcing
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
            pitch=torch.randn(16, 1),
            feats=torch.randn(16, odim),
        )
        if spks > 0:
            inputs["sids"] = torch.randint(0, spks, (1,))
        if langs > 0:
            inputs["lids"] = torch.randint(0, langs, (1,))
        if spk_embed_dim > 0:
            inputs["spembs"] = torch.randn(spk_embed_dim)
        inputs = {
            k: {k2: v2.to(device) for k2, v2 in v.items()}
            if isinstance(v, dict)
            else v.to(device)
            for k, v in inputs.items()
        }
        output_dict = model.inference(**inputs, use_teacher_forcing=True)
        assert output_dict["wav"].size(0) == inputs["feats"].size(0) * upsample_factor
