import pytest
import torch

from espnet2.svs.xiaoice.XiaoiceSing import XiaoiceSing


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
@pytest.mark.parametrize("loss_function", ["FastSpeech1", "XiaoiceSing2"])
@pytest.mark.parametrize("loss_type", ["L1", "L2", "L1+L2"])
@pytest.mark.parametrize(
    "use_masking, use_weighted_masking", [(False, True), (False, False), (True, False)]
)
def test_XiaoiceSing(
    reduction_factor,
    spk_embed_dim,
    spk_embed_integration_type,
    encoder_type,
    decoder_type,
    spks,
    langs,
    loss_function,
    loss_type,
    use_masking,
    use_weighted_masking,
):
    idim = 10
    odim = 5
    model = XiaoiceSing(
        idim=idim,
        odim=odim,
        midi_dim=129,
        duration_dim=10,
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
        spks=spks,
        langs=langs,
        spk_embed_dim=spk_embed_dim,
        spk_embed_integration_type=spk_embed_integration_type,
        use_masking=use_masking,
        use_weighted_masking=use_weighted_masking,
        loss_function=loss_function,
        loss_type=loss_type,
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
    torch.autograd.set_detect_anomaly(True)
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
