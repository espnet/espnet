import pytest
import torch

from espnet2.svs.naive_rnn.naive_rnn_dp import NaiveRNNDP


@pytest.mark.parametrize("eprenet_conv_layers", [0, 1])
@pytest.mark.parametrize("midi_embed_integration_type", ["add", "cat"])
@pytest.mark.parametrize("postnet_layers", [0, 1])
@pytest.mark.parametrize("reduction_factor", [1, 3])
@pytest.mark.parametrize(
    "spk_embed_dim, spk_embed_integration_type",
    [(None, "add"), (2, "add"), (2, "concat")],
)
@pytest.mark.parametrize(
    "spks, langs",
    [(-1, -1), (5, 2)],
)
def test_NaiveRNNDP(
    eprenet_conv_layers,
    midi_embed_integration_type,
    postnet_layers,
    reduction_factor,
    spk_embed_dim,
    spk_embed_integration_type,
    spks,
    langs,
):
    idim = 10
    odim = 4
    model = NaiveRNNDP(
        idim=idim,
        odim=odim,
        midi_dim=129,
        embed_dim=5,
        duration_dim=10,
        eprenet_conv_layers=eprenet_conv_layers,
        eprenet_conv_chans=4,
        eprenet_conv_filts=5,
        elayers=2,
        eunits=6,
        ebidirectional=True,
        midi_embed_integration_type=midi_embed_integration_type,
        dlayers=2,
        dunits=6,
        postnet_layers=postnet_layers,
        postnet_chans=4,
        postnet_filts=5,
        use_batch_norm=True,
        duration_predictor_layers=2,
        duration_predictor_chans=4,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout_rate=0.1,
        reduction_factor=reduction_factor,
        spks=spks,
        langs=langs,
        spk_embed_dim=spk_embed_dim,
        spk_embed_integration_type=spk_embed_integration_type,
        eprenet_dropout_rate=0.2,
        edropout_rate=0.1,
        ddropout_rate=0.1,
        postnet_dropout_rate=0.5,
        init_type="pytorch",
        use_masking=True,
        use_weighted_masking=False,
    )

    inputs = dict(
        text=torch.randint(0, idim, (2, 8)),
        text_lengths=torch.tensor([8, 5], dtype=torch.long),
        feats=torch.randn(2, 16 * reduction_factor, odim),
        feats_lengths=torch.tensor([16, 10], dtype=torch.long) * reduction_factor,
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
        melody_lengths={
            "lab": torch.tensor([8, 5], dtype=torch.long),
            "score": torch.tensor([8, 5], dtype=torch.long),
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
        duration_lengths={
            "lab": torch.tensor([8, 5], dtype=torch.long),
            "score_phn": torch.tensor([8, 5], dtype=torch.long),
            "score_syb": torch.tensor([8, 5], dtype=torch.long),
        },
        slur=torch.randint(0, 2, (2, 8)),
        slur_lengths=torch.tensor([8, 5], dtype=torch.long),
        pitch=torch.randn(2, 16 * reduction_factor, 1),
        pitch_lengths=torch.tensor([16, 10], dtype=torch.long) * reduction_factor,
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
            pitch=torch.randn(11 * reduction_factor, 1),
        )
        if spks > 0:
            inputs["sids"] = torch.randint(0, spks, (1,))
        if langs > 0:
            inputs["lids"] = torch.randint(0, langs, (1,))
        if spk_embed_dim is not None:
            inputs.update(spembs=torch.randn(spk_embed_dim))
        model.inference(**inputs)
