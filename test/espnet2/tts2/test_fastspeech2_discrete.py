import pytest
import torch

from espnet2.tts2.fastspeech2.fastspeech2_discrete import FastSpeech2Discrete


@pytest.mark.parametrize("reduction_factor", [1, 3])
@pytest.mark.parametrize(
    "spk_embed_dim, spk_embed_integration_type",
    [(None, "add"), (2, "add"), (2, "concat")],
)
@pytest.mark.parametrize("encoder_type", ["transformer", "conformer"])
@pytest.mark.parametrize("decoder_type", ["transformer", "conformer"])
@pytest.mark.parametrize(
    "spks, langs",
    [(-1, -1), (5, 2)],
)
def test_fastspeech2discrete(
    reduction_factor,
    spk_embed_dim,
    spk_embed_integration_type,
    encoder_type,
    decoder_type,
    spks,
    langs,
):
    model = FastSpeech2Discrete(
        idim=10,
        odim=5,
        adim=4,
        aheads=2,
        elayers=1,
        eunits=4,
        dlayers=1,
        dunits=4,
        postnet_layers=1,
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
        spks=spks,
        langs=langs,
        spk_embed_dim=spk_embed_dim,
        spk_embed_integration_type=spk_embed_integration_type,
        use_masking=False,
        use_weighted_masking=True,
    )

    inputs = dict(
        text=torch.randint(1, 10, (2, 2)),
        text_lengths=torch.tensor([2, 1], dtype=torch.long),
        discrete_feats=torch.randint(1, 5, (2, 4 * reduction_factor)),
        discrete_feats_lengths=torch.tensor([4, 2], dtype=torch.long)
        * reduction_factor,
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
    if spks > 0:
        inputs.update(sids=torch.randint(0, spks, (2, 1)))
    if langs > 0:
        inputs.update(lids=torch.randint(0, langs, (2, 1)))
    loss, *_ = model(**inputs)
    loss.backward()

    with torch.no_grad():
        model.eval()

        inputs = dict(
            text=torch.randint(0, 10, (2,)),
        )
        if spk_embed_dim is not None:
            inputs.update(spembs=torch.randn(spk_embed_dim))
        if spks > 0:
            inputs.update(sids=torch.randint(0, spks, (1,)))
        if langs > 0:
            inputs.update(lids=torch.randint(0, langs, (1,)))
        model.inference(**inputs)

        # teacher forcing
        inputs.update(durations=torch.tensor([2, 2, 0], dtype=torch.long))
        inputs.update(pitch=torch.tensor([2, 2, 0], dtype=torch.float).unsqueeze(-1))
        inputs.update(energy=torch.tensor([2, 2, 0], dtype=torch.float).unsqueeze(-1))
        model.inference(**inputs, use_teacher_forcing=True)
