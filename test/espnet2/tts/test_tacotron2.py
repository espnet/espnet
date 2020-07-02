import pytest
import torch

from espnet2.tts.tacotron2 import Tacotron2


@pytest.mark.parametrize("econv_layers", [0, 1])
@pytest.mark.parametrize("prenet_layers", [0, 1])
@pytest.mark.parametrize("postnet_layers", [0, 1])
@pytest.mark.parametrize("reduction_factor", [1, 2, 3])
@pytest.mark.parametrize(
    "spk_embed_dim, spk_embed_integration_type",
    [(None, "add"), (2, "add"), (2, "concat")],
)
@pytest.mark.parametrize("loss_type", ["L1+L2", "L1"])
@pytest.mark.parametrize("use_guided_attn_loss", [True, False])
def test_tacotron2(
    econv_layers,
    prenet_layers,
    postnet_layers,
    reduction_factor,
    spk_embed_dim,
    spk_embed_integration_type,
    loss_type,
    use_guided_attn_loss,
):
    model = Tacotron2(
        idim=10,
        odim=5,
        embed_dim=4,
        econv_layers=econv_layers,
        econv_filts=5,
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
        spk_embed_dim=spk_embed_dim,
        spk_embed_integration_type=spk_embed_integration_type,
        loss_type=loss_type,
        use_guided_attn_loss=use_guided_attn_loss,
    )

    inputs = dict(
        text=torch.randint(0, 10, (2, 4)),
        text_lengths=torch.tensor([4, 1], dtype=torch.long),
        speech=torch.randn(2, 3, 5),
        speech_lengths=torch.tensor([3, 1], dtype=torch.long),
    )
    if spk_embed_dim is not None:
        inputs.update(spembs=torch.randn(2, spk_embed_dim))
    loss, *_ = model(**inputs)
    loss.backward()

    with torch.no_grad():
        model.eval()
        inputs = dict(text=torch.randint(0, 10, (2,)),)
        if spk_embed_dim is not None:
            inputs.update(spembs=torch.randn(spk_embed_dim))
        model.inference(**inputs, maxlenratio=1.0)
