import pytest
import torch

from espnet2.tok.hifigan_loss import HiFiGANReconstructionLoss
from espnet2.tok.output import SpeechTokenizerOutput

TINY_GENERATOR_PARAMS = {
    "channels": 8,
    "upsample_scales": [2],
    "upsample_kernel_sizes": [4],
    "resblock_kernel_sizes": [3],
    "resblock_dilations": [[1, 3, 5]],
    "use_weight_norm": False,
}

TINY_DISCRIMINATOR_PARAMS = {
    "scales": 1,
    "scale_discriminator_params": {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_sizes": [3, 3, 3, 3],
        "channels": 4,
        "max_downsample_channels": 16,
        "max_groups": 1,
        "bias": True,
        "downsample_scales": [1, 1],
        "nonlinear_activation": "LeakyReLU",
        "nonlinear_activation_params": {"negative_slope": 0.1},
        "use_weight_norm": False,
        "use_spectral_norm": False,
    },
    "follow_official_norm": False,
    "periods": [2],
    "period_discriminator_params": {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_sizes": [3, 3],
        "channels": 4,
        "downsample_scales": [1, 1],
        "max_downsample_channels": 16,
        "bias": True,
        "nonlinear_activation": "LeakyReLU",
        "nonlinear_activation_params": {"negative_slope": 0.1},
        "use_weight_norm": False,
        "use_spectral_norm": False,
    },
}


def make_loss(cache_generator_outputs=True, **kwargs):
    loss_params = {
        "use_feat_match_loss": False,
        "use_mel_loss": False,
        "cache_generator_outputs": cache_generator_outputs,
    }
    loss_params.update(kwargs)
    return HiFiGANReconstructionLoss(
        num_tokens=4,
        token_embed_dim=3,
        speaker_embed_dim=2,
        segment_size=4,
        generator_params=TINY_GENERATOR_PARAMS,
        discriminator_params=TINY_DISCRIMINATOR_PARAMS,
        **loss_params,
    )


def make_tokenizer_output():
    assignment = torch.nn.functional.one_hot(
        torch.tensor([[0, 1, 2, 3, 0]]), num_classes=4
    ).float()
    assignment.requires_grad_()
    return SpeechTokenizerOutput(
        continuous=torch.zeros(1, 5, 2),
        assignment=assignment,
        token_ids=assignment.argmax(dim=-1),
        lengths=torch.tensor([5]),
    )


def test_synthesize():
    loss_module = make_loss()
    tokenizer_output = make_tokenizer_output()
    spembs = torch.ones(1, 2)

    condition = loss_module._prepare_condition(tokenizer_output, spembs)
    token_features = torch.matmul(
        tokenizer_output.assignment.detach(), loss_module.embedding.embed.weight.detach()
    )
    expected_condition = torch.cat(
        [token_features, spembs.unsqueeze(1).expand(-1, 5, -1)], dim=-1
    ).transpose(1, 2)
    torch.testing.assert_close(condition.detach(), expected_condition)

    waveform, waveform_lengths = loss_module.synthesize(tokenizer_output, spembs=spembs)

    assert waveform.shape == (1, 1, 10)
    torch.testing.assert_close(waveform_lengths, torch.tensor([10]))


def test_generator_and_discriminator_losses_with_cache():
    loss_module = make_loss()
    tokenizer_output = make_tokenizer_output()
    speech = torch.randn(1, 10)
    spembs = torch.randn(1, 2)

    generator_loss, _, _ = loss_module(tokenizer_output, speech, spembs)
    generator_loss.backward()
    assert tokenizer_output.assignment.grad is not None
    assert torch.count_nonzero(tokenizer_output.assignment.grad) > 0
    assert loss_module._cache is not None

    for parameter in loss_module.parameters():
        parameter.grad = None
    discriminator_loss, _, _ = loss_module.forward_discriminator(
        tokenizer_output, speech, spembs
    )
    discriminator_loss.backward()

    assert loss_module._cache is None
    assert any(p.grad is not None for p in loss_module.discriminator.parameters())
    assert all(p.grad is None for p in loss_module.generator.parameters())
    assert all(p.grad is None for p in loss_module.embedding.parameters())


def test_generator_loss_with_auxiliary_losses():
    loss_module = make_loss(
        use_feat_match_loss=True,
        use_mel_loss=True,
        mel_loss_params={
            "fs": 16000,
            "n_fft": 4,
            "hop_length": 1,
            "win_length": 4,
            "n_mels": 2,
        },
    )
    tokenizer_output = make_tokenizer_output()
    speech = torch.randn(1, 10)
    spembs = torch.randn(1, 2)

    loss, stats, _ = loss_module(tokenizer_output, speech, spembs)
    speech_hat, speech_real, _ = loss_module._cache
    fake_outputs = loss_module.discriminator(speech_hat)
    real_outputs = loss_module.discriminator(speech_real)
    expected_adv_loss = loss_module.generator_adv_loss(fake_outputs)
    expected_feat_match_loss = 2.0 * loss_module.feat_match_loss(
        fake_outputs, real_outputs
    )
    expected_mel_loss = 45.0 * loss_module.mel_loss(speech_hat, speech_real)

    torch.testing.assert_close(stats["adv_loss"].squeeze(), expected_adv_loss)
    torch.testing.assert_close(
        stats["feat_match_loss"].squeeze(), expected_feat_match_loss
    )
    torch.testing.assert_close(stats["mel_loss"].squeeze(), expected_mel_loss)
    torch.testing.assert_close(
        loss.squeeze(),
        expected_adv_loss + expected_feat_match_loss + expected_mel_loss,
    )


def test_generator_cache_rejects_second_generator_forward():
    loss_module = make_loss()
    tokenizer_output = make_tokenizer_output()
    speech = torch.randn(1, 10)
    spembs = torch.randn(1, 2)

    loss_module(tokenizer_output, speech, spembs)

    with pytest.raises(RuntimeError, match="already cached"):
        loss_module(tokenizer_output, speech, spembs)


def test_generator_cache_rejects_different_discriminator_batch():
    loss_module = make_loss()
    tokenizer_output = make_tokenizer_output()
    speech = torch.randn(1, 10)
    different_speech = torch.randn(1, 10)
    spembs = torch.randn(1, 2)

    loss_module(tokenizer_output, speech, spembs)

    with pytest.raises(RuntimeError, match="different speech tensor"):
        loss_module.forward_discriminator(tokenizer_output, different_speech, spembs)
    assert loss_module._cache is None
