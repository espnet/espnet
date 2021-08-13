# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test VITS related modules."""

from distutils.version import LooseVersion

import pytest
import torch

from espnet2.gan_tts.vits.generator import VITSGenerator


def make_generator_args(**kwargs):
    defaults = {
        "vocabs": 10,
        "aux_channels": 5,
        "hidden_channels": 4,
        "spks": -1,
        "global_channels": -1,
        "segment_size": 4,
        "text_encoder_attention_heads": 2,
        "text_encoder_attention_expand": 4,
        "text_encoder_blocks": 2,
        "text_encoder_kernel_size": 3,
        "text_encoder_dropout_rate": 0.1,
        "text_encoder_positional_dropout_rate": 0.0,
        "text_encoder_attention_dropout_rate": 0.0,
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
        "stochastic_duration_predictor_kernel_size": 3,
        "stochastic_duration_predictor_dropout_rate": 0.5,
        "stochastic_duration_predictor_flows": 2,
        "stochastic_duration_predictor_dds_conv_layers": 3,
    }
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
@torch.no_grad()
def test_vits_generator_forward():
    idim = 10
    odim = 5
    args = make_generator_args(vocabs=idim, aux_channels=odim)
    model = VITSGenerator(**args)

    # check forward
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, odim, 16),
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


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion("1.4"),
    reason="Pytorch >= 1.4 is required.",
)
@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@torch.no_grad()
def test_multi_speaker_vits_generator_forward():
    idim = 10
    odim = 5
    spks = 10
    global_channels = 8
    args = make_generator_args(
        vocabs=idim,
        aux_channels=odim,
        spks=spks,
        global_channels=global_channels,
    )
    model = VITSGenerator(**args)

    # check forward
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, odim, 16),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        sids=torch.randint(0, spks, (2,)),
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
        sids=torch.randint(0, spks, (1,)),
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
        sids=torch.randint(0, spks, (1,)),
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
