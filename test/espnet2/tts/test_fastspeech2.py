import pytest
import torch

from espnet2.tts.fastspeech2 import FastSpeech2


@pytest.mark.parametrize("postnet_layers", [0, 1])
@pytest.mark.parametrize("reduction_factor", [1, 2, 3])
@pytest.mark.parametrize(
    "spk_embed_dim, spk_embed_integration_type",
    [(None, "add"), (2, "add"), (2, "concat")],
)
@pytest.mark.parametrize("use_gst", [True, False])
@pytest.mark.parametrize("encoder_type", ["transformer", "conformer"])
@pytest.mark.parametrize("decoder_type", ["transformer", "conformer"])
@pytest.mark.parametrize(
    "use_masking, use_weighted_masking", [[True, False], [False, True]]
)
def test_fastspeech2(
    postnet_layers,
    reduction_factor,
    spk_embed_dim,
    spk_embed_integration_type,
    encoder_type,
    decoder_type,
    use_gst,
    use_masking,
    use_weighted_masking,
):
    model = FastSpeech2(
        idim=10,
        odim=5,
        adim=4,
        aheads=2,
        elayers=1,
        eunits=4,
        dlayers=1,
        dunits=4,
        postnet_layers=postnet_layers,
        postnet_chans=4,
        postnet_filts=5,
        reduction_factor=reduction_factor,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        duration_predictor_layers=2,
        duration_predictor_chans=4,
        duration_predictor_kernel_size=3,
        energy_predictor_layers=2,
        energy_predictor_chans=4,
        energy_predictor_kernel_size=3,
        energy_predictor_dropout=0.5,
        energy_embed_kernel_size=9,
        energy_embed_dropout=0.5,
        pitch_predictor_layers=2,
        pitch_predictor_chans=4,
        pitch_predictor_kernel_size=3,
        pitch_predictor_dropout=0.5,
        pitch_embed_kernel_size=9,
        pitch_embed_dropout=0.5,
        spk_embed_dim=spk_embed_dim,
        spk_embed_integration_type=spk_embed_integration_type,
        use_gst=use_gst,
        gst_tokens=2,
        gst_heads=4,
        gst_conv_layers=2,
        gst_conv_chans_list=[2, 4],
        gst_conv_kernel_size=3,
        gst_conv_stride=2,
        gst_gru_layers=1,
        gst_gru_units=4,
        use_masking=use_masking,
        use_weighted_masking=use_weighted_masking,
    )

    inputs = dict(
        text=torch.randint(1, 10, (2, 2)),
        text_lengths=torch.tensor([2, 1], dtype=torch.long),
        speech=torch.randn(2, 4 * reduction_factor, 5),
        speech_lengths=torch.tensor([4, 2], dtype=torch.long) * reduction_factor,
        durations=torch.tensor([[2, 2, 0], [2, 0, 0]], dtype=torch.long),
        pitch=torch.tensor([[2, 2, 0], [2, 0, 0]], dtype=torch.float).unsqueeze(-1),
        energy=torch.tensor([[2, 2, 0], [2, 0, 0]], dtype=torch.float).unsqueeze(-1),
        # NOTE(kan-bayashi): +1 for eos
        durations_lengths=torch.tensor([2 + 1, 1 + 1], dtype=torch.long),
        pitch_lengths=torch.tensor([2 + 1, 1 + 1], dtype=torch.long),
        energy_lengths=torch.tensor([2 + 1, 1 + 1], dtype=torch.long),
    )
    if spk_embed_dim is not None:
        inputs.update(spembs=torch.randn(2, spk_embed_dim))
    loss, *_ = model(**inputs)
    loss.backward()

    with torch.no_grad():
        model.eval()

        inputs = dict(
            text=torch.randint(0, 10, (2,)),
        )
        if use_gst:
            inputs.update(speech=torch.randn(5, 5))
        if spk_embed_dim is not None:
            inputs.update(spembs=torch.randn(spk_embed_dim))
        model.inference(**inputs)

        # teacher forcing
        inputs.update(durations=torch.tensor([2, 2, 0], dtype=torch.long))
        inputs.update(pitch=torch.tensor([2, 2, 0], dtype=torch.float).unsqueeze(-1))
        inputs.update(energy=torch.tensor([2, 2, 0], dtype=torch.float).unsqueeze(-1))
        model.inference(**inputs, use_teacher_forcing=True)
