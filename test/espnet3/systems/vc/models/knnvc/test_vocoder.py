"""Tests for the kNN-VC HiFi-GAN training model."""

from test.espnet3.systems.vc.models.knnvc.tiny import (
    TINY_DISCRIMINATOR,
    TINY_GENERATOR,
    TINY_HOP,
    TINY_MEL,
)

import torch

from espnet3.components.modeling.optimization_spec import OptimizationStep
from espnet3.systems.vc.models.knnvc.vocoder import (
    DEFAULT_GENERATOR_PARAMS,
    KNNVCGenerator,
    KNNVCVocoderModel,
    LogMelSpectrogram,
    build_generator_params,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_build_generator_params_overrides       | Overrides merge into the     |
# |                                             | kNN-VC defaults.             |
# | test_build_generator_params_does_not_share_nested_defaults                 |
# |                          | Returned nested lists are copies, so the        |
# |                          | module-level defaults cannot be mutated.        |
# | test_generator_shapes                       | forward/inference upsample   |
# |                                             | by prod(upsample_scales).    |
# | test_log_mel_frame_count                    | frames == samples / hop.     |
# | test_forward_returns_two_optimization_steps | discriminator then generator |
# |                                             | steps, stats and weight.     |
# | test_forward_supports_manual_gan_updates    | Backward D, step D, then     |
# |                          | backward G succeeds (no in-place graph errors)  |
# |                          | and D gets no gradient from the generator loss. |
# | test_param_selectors_partition_parameters   | Every trainable param is in  |
# |                          | exactly one of `generator` / `discriminator`.   |
# | test_period_discriminators_use_official_output_kernel                      |
# |                          | The period discriminators end in the official   |
# |                          | (3, 1) convolution, as HiFi-GAN does.            |
# | test_official_output_kernel_can_be_disabled                                |
# |                          | Opting out leaves espnet2's stock (2, 1).       |


def _model():
    torch.manual_seed(0)
    return KNNVCVocoderModel(
        generator=TINY_GENERATOR, discriminator=TINY_DISCRIMINATOR, mel=TINY_MEL
    )


def _batch(batch_size=2, frames=12):
    feats = torch.randn(batch_size, frames, TINY_GENERATOR["in_channels"])
    speech = torch.rand(batch_size, frames * TINY_HOP) * 2 - 1
    lengths = torch.full((batch_size,), frames)
    return {
        "feats": feats,
        "feats_lengths": lengths,
        "speech": speech,
        "speech_lengths": lengths * TINY_HOP,
    }


def test_build_generator_params_overrides():
    params = build_generator_params({"channels": 64})
    assert params["channels"] == 64
    assert params["in_channels"] == DEFAULT_GENERATOR_PARAMS["in_channels"] == 1024
    assert params["projection_channels"] == 512
    assert build_generator_params(None) == DEFAULT_GENERATOR_PARAMS


def test_build_generator_params_does_not_share_nested_defaults():
    """Each call gets its own nested lists, not the module-level objects.

    ``DEFAULT_GENERATOR_PARAMS`` holds lists (``upsample_scales``,
    ``resblock_dilations``); a shallow copy would let one model's in-place edit
    change the defaults for every model built afterwards.
    """
    first = build_generator_params()
    second = build_generator_params()
    assert first["upsample_scales"] is not second["upsample_scales"]
    assert first["upsample_scales"] is not DEFAULT_GENERATOR_PARAMS["upsample_scales"]

    first["upsample_scales"].append(99)
    first["resblock_dilations"][0].append(99)
    assert DEFAULT_GENERATOR_PARAMS["upsample_scales"] == [10, 8, 2, 2]
    assert DEFAULT_GENERATOR_PARAMS["resblock_dilations"][0] == [1, 3, 5]
    assert second["upsample_scales"] == [10, 8, 2, 2]


def test_generator_shapes():
    generator = KNNVCGenerator(**TINY_GENERATOR)
    assert generator.upsample_factor == TINY_HOP
    feats = torch.randn(3, 10, TINY_GENERATOR["in_channels"])
    assert generator(feats).shape == (3, 1, 10 * TINY_HOP)
    assert generator.inference(feats[0]).shape == (10 * TINY_HOP,)
    generator.remove_weight_norm()
    assert generator(feats).shape == (3, 1, 10 * TINY_HOP)


def test_log_mel_frame_count():
    mel = LogMelSpectrogram(**TINY_MEL)
    wav = torch.randn(2, 20 * TINY_HOP)
    out = mel(wav)
    assert out.shape == (2, TINY_MEL["n_mels"], 20)
    assert torch.isfinite(out).all()


def test_forward_returns_two_optimization_steps():
    model = _model()
    steps, stats, weight = model(**_batch())

    assert [step.name for step in steps] == ["discriminator", "generator"]
    assert all(isinstance(step, OptimizationStep) for step in steps)
    assert all(step.loss.requires_grad and step.loss.dim() == 0 for step in steps)
    assert set(stats) >= {
        "mel_loss",
        "generator_adv_loss",
        "feat_match_loss",
        "generator_loss",
        "discriminator_loss",
    }
    assert weight.item() == 2
    assert model.hop_length == TINY_HOP


def test_forward_supports_manual_gan_updates():
    model = _model()
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)

    steps, _, _ = model(**_batch())
    d_step, g_step = steps
    # Same order as ESPnetLightningModule._run_multi_optimizer_updates:
    # discriminator first, its parameters updated in place, then generator.
    optimizer_d.zero_grad()
    d_step.loss.backward()
    optimizer_d.step()
    optimizer_g.zero_grad()
    g_step.loss.backward()
    optimizer_g.step()

    disc_grads = [p.grad for p in model.discriminator.parameters() if p.requires_grad]
    gen_grads = [p.grad for p in model.generator.parameters() if p.requires_grad]
    assert all(g is not None for g in gen_grads)
    # The generator loss must not leak gradients into the discriminator.
    optimizer_d.zero_grad()
    steps, _, _ = model(**_batch())
    steps[1].loss.backward()
    assert all(
        p.grad is None or torch.all(p.grad == 0)
        for p in model.discriminator.parameters()
    )
    assert all(p.requires_grad for p in model.discriminator.parameters())
    assert len(disc_grads) > 0


def test_param_selectors_partition_parameters():
    model = _model()
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    generator = {n for n in names if n.startswith("generator.")}
    discriminator = {n for n in names if n.startswith("discriminator.")}
    assert generator and discriminator
    assert generator | discriminator == set(names)
    # The mel-spectrogram transform contributes buffers only.
    assert not [n for n in names if n.startswith("mel_spectrogram.")]


def test_period_discriminators_use_official_output_kernel():
    """The period discriminators match HiFi-GAN's ``DiscriminatorP``.

    ``espnet2`` derives the output kernel as ``kernel_sizes[1] - 1`` and asserts
    that ``kernel_sizes[1]`` is odd, so the official ``(3, 1)`` kernel is not
    reachable through configuration. ``KNNVCVocoderModel`` rebuilds it, which is
    what kNN-VC's HiFi-GAN is trained with.
    """
    model = KNNVCVocoderModel(
        generator=TINY_GENERATOR, discriminator=TINY_DISCRIMINATOR
    )
    for period_discriminator in model.discriminator.mpd.discriminators:
        weight = dict(period_discriminator.named_parameters())["output_conv.weight_v"]
        assert tuple(weight.shape[-2:]) == (3, 1)
        assert period_discriminator.output_conv.padding == (1, 0)


def test_official_output_kernel_can_be_disabled():
    """Opting out leaves the stock ``espnet2`` ``(2, 1)`` output kernel."""
    model = KNNVCVocoderModel(
        generator=TINY_GENERATOR,
        discriminator=TINY_DISCRIMINATOR,
        official_period_output_kernel=False,
    )
    for period_discriminator in model.discriminator.mpd.discriminators:
        weight = dict(period_discriminator.named_parameters())["output_conv.weight_v"]
        assert tuple(weight.shape[-2:]) == (2, 1)
