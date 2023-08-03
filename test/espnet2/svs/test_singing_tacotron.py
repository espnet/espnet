import pytest
import torch

from espnet2.svs.singing_tacotron.singing_tacotron import singing_tacotron


@pytest.mark.parametrize("prenet_layers", [0, 1])
@pytest.mark.parametrize("postnet_layers", [0, 1])
@pytest.mark.parametrize("reduction_factor", [1, 3])
@pytest.mark.parametrize("atype", ["location", "forward", "forward_ta", "GDCA"])
@pytest.mark.parametrize(
    "spk_embed_dim, spk_embed_integration_type",
    [(None, "add"), (2, "add"), (2, "concat")],
)
@pytest.mark.parametrize(
    "spks, langs, use_gst",
    [(-1, -1, False), (5, 2, True)],
)
@pytest.mark.parametrize("loss_type", ["L1", "L2", "L1+L2"])
@pytest.mark.parametrize("use_guided_attn_loss", [True, False])
def test_singing_tacotron(
    prenet_layers,
    postnet_layers,
    reduction_factor,
    loss_type,
    atype,
    spks,
    langs,
    spk_embed_dim,
    spk_embed_integration_type,
    use_gst,
    use_guided_attn_loss,
):
    idim = 10
    odim = 5
    model = singing_tacotron(
        idim=idim,
        odim=odim,
        midi_dim=129,
        embed_dim=5,
        duration_dim=10,
        elayers=1,
        eunits=4,
        econv_layers=1,
        econv_filts=5,
        econv_chans=4,
        atype=atype,
        adim=4,
        dlayers=1,
        dunits=4,
        prenet_layers=prenet_layers,
        prenet_units=4,
        postnet_layers=postnet_layers,
        postnet_chans=4,
        postnet_filts=5,
        reduction_factor=reduction_factor,
        spks=spks,
        langs=langs,
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
        loss_type=loss_type,
        use_guided_attn_loss=use_guided_attn_loss,
        guided_attn_loss_sigma=0.4,
        guided_attn_loss_lambda=1.0,
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
        if use_gst:
            inputs.update(feats=torch.randn(5, 5))
        if atype == "GDCA":
            use_dynamic_filter = True
            use_att_constraint = False
        else:
            use_dynamic_filter = False
            use_att_constraint = True
        model.inference(
            **inputs,
            maxlenratio=1.0,
            use_att_constraint=use_att_constraint,
            use_dynamic_filter=use_dynamic_filter
        )

        # teacher forcing
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
            pitch=torch.randn(11 * reduction_factor, 1),
            feats=torch.randn(11 * reduction_factor, odim),
        )
        if spks > 0:
            inputs["sids"] = torch.randint(0, spks, (1,))
        if langs > 0:
            inputs["lids"] = torch.randint(0, langs, (1,))
        if spk_embed_dim is not None:
            inputs.update(spembs=torch.randn(spk_embed_dim))
        if use_gst:
            inputs.update(feats=torch.randn(5, 5))
        use_dynamic_filter = False
        use_att_constraint = False

        model.inference(
            **inputs,
            use_teacher_forcing=True,
            use_att_constraint=use_att_constraint,
            use_dynamic_filter=use_dynamic_filter
        )
