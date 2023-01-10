# Copyright 2021 Tomoki Hayashi
# Copyright 2023 Yifeng Yu
#  Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Test VISinger generator modules."""

import pytest
import torch

from espnet2.gan_svs.vits.generator import VITSGenerator


def make_generator_args(**kwargs):
    defaults = dict(
        vocabs=10,
        midi_dim=129,
        beat_dim=600,
        aux_channels=5,
        hidden_channels=4,
        spks=-1,
        langs=-1,
        spk_embed_dim=-1,
        global_channels=-1,
        segment_size=4,
        text_encoder_attention_heads=2,
        text_encoder_ffn_expand=4,
        text_encoder_blocks=2,
        text_encoder_positionwise_layer_type="conv1d",
        text_encoder_positionwise_conv_kernel_size=1,
        text_encoder_normalize_before=True,
        text_encoder_dropout_rate=0.1,
        text_encoder_positional_dropout_rate=0.0,
        text_encoder_attention_dropout_rate=0.0,
        text_encoder_conformer_kernel_size=7,
        use_macaron_style_in_text_encoder=True,
        use_conformer_conv_in_text_encoder=True,
        decoder_kernel_size=7,
        decoder_channels=16,
        decoder_upsample_scales=[16, 16],
        decoder_upsample_kernel_sizes=[32, 32],
        decoder_resblock_kernel_sizes=[3, 5],
        decoder_resblock_dilations=[[1, 3], [1, 3]],
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
        use_dp=True,
        use_visinger=True,
    )
    defaults.update(kwargs)
    return defaults


# NOTE(kan-bayashi): first forward requires jit compile
#   so a little bit more time is needed to run. Therefore,
#   here we extend execution timeout from 2 sec to 5 sec.
@pytest.mark.execution_timeout(5)
@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@torch.no_grad()
@pytest.mark.parametrize(
    "model_dict",
    [
        ({}),
        ({"text_encoder_positionwise_layer_type": "linear"}),
        ({"text_encoder_positionwise_layer_type": "conv1d-linear"}),
        ({"text_encoder_normalize_before": False}),
        ({"use_macaron_style_in_text_encoder": False}),
        ({"use_conformer_conv_in_text_encoder": False}),
        (
            {
                "text_encoder_positional_encoding_layer_type": "scaled_abs_pos",
                "text_encoder_self_attention_layer_type": "selfattn",
            }
        ),
        ({"spk_embed_dim": 16, "global_channels": 4}),
        ({"langs": 16, "global_channels": 4}),
    ],
)
def test_vits_generator_forward(model_dict):
    idim = 10
    odim = 5
    args = make_generator_args(vocabs=idim, aux_channels=odim, **model_dict)
    model = VITSGenerator(**args)

    # check forward
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, odim, 16),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        label=torch.randint(0, idim, (2, 8)),
        label_lengths=torch.tensor([8, 5], dtype=torch.long),
        melody=torch.randint(0, 127, (2, 8)),
        melody_lengths=torch.tensor([8, 5], dtype=torch.long),
        tempo=torch.randint(1, idim, (2, 8)),
        tempo_lengths=torch.tensor([8, 5], dtype=torch.long),
        beat=torch.randint(1, idim, (2, 8)).float(),
        beat_lengths=torch.tensor([8, 5], dtype=torch.long),
        pitch=torch.randn(2, 16, 1),
        pitch_lengths=torch.tensor([16, 13], dtype=torch.long),
        duration=torch.tensor(
            [[1, 2, 2, 3, 1, 3, 2, 2], [2, 2, 1, 4, 1, 2, 1, 3]], dtype=torch.int64
        ),
    )
    if args["spk_embed_dim"] > 0:
        inputs["spembs"] = torch.randn(2, args["spk_embed_dim"])
    if args["langs"] > 0:
        inputs["lids"] = torch.randint(0, args["langs"], (2, 1))
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
        label=torch.randint(
            0,
            idim,
            (
                2,
                5,
            ),
        ),
        label_lengths=torch.tensor([5, 3], dtype=torch.long),
        melody=torch.randint(
            0,
            127,
            (
                2,
                5,
            ),
        ),
        tempo=torch.randint(
            1,
            idim,
            (
                2,
                5,
            ),
        ),
        beat=torch.randint(
            1,
            idim,
            (
                2,
                5,
            ),
        ).float(),
        pitch=torch.randn(2, 16, 1),
    )
    if args["spk_embed_dim"] > 0:
        inputs["spembs"] = torch.randn(args["spk_embed_dim"])
    if args["langs"] > 0:
        inputs["lids"] = torch.randint(0, args["langs"], (1,))
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
        label=torch.randint(
            0,
            idim,
            (
                1,
                5,
            ),
        ),
        label_lengths=torch.tensor([5], dtype=torch.long),
        melody=torch.randint(
            0,
            127,
            (
                1,
                5,
            ),
        ),
        tempo=torch.randint(
            1,
            idim,
            (
                1,
                5,
            ),
        ),
        beat=torch.randint(
            1,
            idim,
            (
                1,
                5,
            ),
        ).float(),
        pitch=torch.randn(1, 16, 1),
        feats=torch.randn(1, odim, 16),
        feats_lengths=torch.tensor([16], dtype=torch.long),
    )
    if args["spk_embed_dim"] > 0:
        inputs["spembs"] = torch.randn(args["spk_embed_dim"])
    if args["langs"] > 0:
        inputs["lids"] = torch.randint(0, args["langs"], (1,))
    output = model.inference(**inputs, use_teacher_forcing=True)
    assert output.size(1) == inputs["feats"].size(2) * model.upsample_factor


@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@torch.no_grad()
@pytest.mark.parametrize(
    "model_dict",
    [
        ({}),
        ({"text_encoder_positionwise_layer_type": "linear"}),
        ({"text_encoder_positionwise_layer_type": "conv1d-linear"}),
        ({"text_encoder_normalize_before": False}),
        ({"use_macaron_style_in_text_encoder": False}),
        ({"use_conformer_conv_in_text_encoder": False}),
        (
            {
                "text_encoder_positional_encoding_layer_type": "scaled_abs_pos",
                "text_encoder_self_attention_layer_type": "selfattn",
            }
        ),
        ({"spk_embed_dim": 16}),
        ({"langs": 16}),
    ],
)
def test_multi_speaker_vits_generator_forward(model_dict):
    idim = 10
    odim = 5
    spks = 10
    global_channels = 8
    args = make_generator_args(
        vocabs=idim,
        aux_channels=odim,
        spks=spks,
        global_channels=global_channels,
        **model_dict,
    )
    model = VITSGenerator(**args)

    # check forward
    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, odim, 16),
        feats_lengths=torch.tensor([16, 13], dtype=torch.long),
        label=torch.randint(0, idim, (2, 8)),
        label_lengths=torch.tensor([8, 5], dtype=torch.long),
        melody=torch.randint(0, 127, (2, 8)),
        melody_lengths=torch.tensor([8, 5], dtype=torch.long),
        tempo=torch.randint(1, idim, (2, 8)),
        tempo_lengths=torch.tensor([8, 5], dtype=torch.long),
        beat=torch.randint(1, idim, (2, 8)).float(),
        beat_lengths=torch.tensor([8, 5], dtype=torch.long),
        pitch=torch.randn(2, 16, 1),
        pitch_lengths=torch.tensor([16, 13], dtype=torch.long),
        duration=torch.tensor(
            [[1, 2, 2, 3, 1, 3, 2, 2], [2, 2, 1, 4, 1, 2, 1, 3]], dtype=torch.int64
        ),
        sids=torch.randint(0, spks, (2,)),
    )
    if args["spk_embed_dim"] > 0:
        inputs["spembs"] = torch.randn(2, args["spk_embed_dim"])
    if args["langs"] > 0:
        inputs["lids"] = torch.randint(0, args["langs"], (2, 1))
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
        label=torch.randint(
            0,
            idim,
            (
                2,
                5,
            ),
        ),
        label_lengths=torch.tensor([5, 3], dtype=torch.long),
        melody=torch.randint(
            0,
            127,
            (
                2,
                5,
            ),
        ),
        tempo=torch.randint(
            1,
            idim,
            (
                2,
                5,
            ),
        ),
        beat=torch.randint(
            1,
            idim,
            (
                2,
                5,
            ),
        ).float(),
        pitch=torch.randn(2, 16, 1),
        sids=torch.randint(0, spks, (1,)),
    )
    if args["spk_embed_dim"] > 0:
        inputs["spembs"] = torch.randn(args["spk_embed_dim"])
    if args["langs"] > 0:
        inputs["lids"] = torch.randint(0, args["langs"], (1,))
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
        feats=torch.randn(1, odim, 16),
        feats_lengths=torch.tensor([16], dtype=torch.long),
        label=torch.randint(
            0,
            idim,
            (
                1,
                5,
            ),
        ),
        label_lengths=torch.tensor([5], dtype=torch.long),
        melody=torch.randint(
            0,
            127,
            (
                1,
                5,
            ),
        ),
        tempo=torch.randint(
            1,
            idim,
            (
                1,
                5,
            ),
        ),
        beat=torch.randint(
            1,
            idim,
            (
                1,
                5,
            ),
        ).float(),
        pitch=torch.randn(1, 16, 1),
        sids=torch.randint(0, spks, (1,)),
    )
    if args["spk_embed_dim"] > 0:
        inputs["spembs"] = torch.randn(args["spk_embed_dim"])
    if args["langs"] > 0:
        inputs["lids"] = torch.randint(0, args["langs"], (1,))
    output = model.inference(**inputs, use_teacher_forcing=True)
    assert output.size(1) == inputs["feats"].size(2) * model.upsample_factor
