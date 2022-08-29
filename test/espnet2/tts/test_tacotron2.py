import pytest
import torch

from espnet2.tts.tacotron2 import Tacotron2


@pytest.mark.parametrize("prenet_layers", [0, 1])
@pytest.mark.parametrize("postnet_layers", [0, 1])
@pytest.mark.parametrize("reduction_factor", [1, 3])
@pytest.mark.parametrize(
    "spk_embed_dim, spk_embed_integration_type",
    [(None, "add"), (2, "add"), (2, "concat")],
)
@pytest.mark.parametrize(
    "spks, langs, use_gst", [(-1, -1, False), (5, 2, True)],
)
@pytest.mark.parametrize("use_guided_attn_loss", [True, False])
def test_tacotron2(
    prenet_layers,
    postnet_layers,
    reduction_factor,
    spks,
    langs,
    spk_embed_dim,
    spk_embed_integration_type,
    use_gst,
    use_guided_attn_loss,
):
    model = Tacotron2(
        idim=10,
        odim=5,
        adim=4,
        embed_dim=4,
        econv_layers=1,
        econv_filts=5,
        econv_chans=4,
        elayers=1,
        eunits=4,
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
        loss_type="L1+L2",
        use_guided_attn_loss=use_guided_attn_loss,
    )

    inputs = dict(
        text=torch.randint(0, 10, (2, 4)),
        text_lengths=torch.tensor([4, 1], dtype=torch.long),
        feats=torch.randn(2, 5, 5),
        feats_lengths=torch.tensor([5, 3], dtype=torch.long),
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

        # free running
        inputs = dict(text=torch.randint(0, 10, (2,)),)
        if use_gst:
            inputs.update(feats=torch.randn(5, 5))
        if spk_embed_dim is not None:
            inputs.update(spembs=torch.randn(spk_embed_dim))
        if spks > 0:
            inputs.update(sids=torch.randint(0, spks, (1,)))
        if langs > 0:
            inputs.update(lids=torch.randint(0, langs, (1,)))
        model.inference(**inputs, maxlenratio=1.0)

        # teacher forcing
        inputs.update(feats=torch.randn(5, 5))
        model.inference(**inputs, use_teacher_forcing=True)
