import pytest
import torch
from packaging.version import parse as V

from espnet2.tts.prodiff import ProDiff
from espnet2.tts.prodiff.loss import SSimLoss

is_torch_1_7_plus = V(torch.__version__) >= V("1.7.0")


@pytest.mark.parametrize("reduction_factor", [1])
@pytest.mark.parametrize(
    "spk_embed_dim, spk_embed_integration_type",
    [(None, "add"), (2, "add"), (2, "concat")],
)
@pytest.mark.parametrize("encoder_type", ["transformer", "conformer"])
@pytest.mark.parametrize("diffusion_scheduler", ["linear", "cosine", "vpsde"])
@pytest.mark.parametrize(
    "spks, langs, use_gst",
    [(-1, -1, False), (5, 2, True)],
)
@pytest.mark.skipif(
    not is_torch_1_7_plus,
    reason="Pytorch >= 1.7 is required.",
)
def test_prodiff(
    reduction_factor,
    spk_embed_dim,
    spk_embed_integration_type,
    diffusion_scheduler,
    encoder_type,
    use_gst,
    spks,
    langs,
):
    model = ProDiff(
        idim=10,
        odim=5,
        adim=4,
        aheads=2,
        elayers=1,
        eunits=4,
        reduction_factor=reduction_factor,
        encoder_type=encoder_type,
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
        use_gst=use_gst,
        gst_tokens=2,
        gst_heads=4,
        gst_conv_layers=2,
        gst_conv_chans_list=[2, 4],
        gst_conv_kernel_size=3,
        gst_conv_stride=2,
        gst_gru_layers=1,
        gst_gru_units=4,
        use_masking=False,
        use_weighted_masking=True,
        denoiser_layers=1,
        denoiser_channels=4,
        diffusion_steps=1,
        diffusion_scheduler=diffusion_scheduler,
    )

    inputs = dict(
        text=torch.randint(1, 10, (2, 2)),
        text_lengths=torch.tensor([2, 1], dtype=torch.long),
        feats=torch.randn(2, 4 * reduction_factor, 5),
        feats_lengths=torch.tensor([4, 2], dtype=torch.long) * reduction_factor,
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
            feats=torch.randn(2, 4 * reduction_factor, 5),
            durations=torch.tensor([[2, 2, 0], [2, 0, 0]], dtype=torch.long),
        )
        if use_gst:
            inputs.update(feats=torch.randn(5, 5))
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


@pytest.mark.parametrize("reduction_type", ["none", "mean"])
@pytest.mark.skipif(
    not is_torch_1_7_plus,
    reason="Pytorch >= 1.7 is required.",
)
def test_ssim(reduction_type):
    lossfun = SSimLoss(reduction=reduction_type)
    feats = torch.randn(2, 4, 5)
    lossfun(feats, feats)
