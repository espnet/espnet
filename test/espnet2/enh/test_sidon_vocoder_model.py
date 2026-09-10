import pytest
import torch
from torch import nn

from espnet2.enh.decoder.sidon_vocoder import SidonVocoder
from espnet2.enh.sidon_vocoder_model import SidonVocoderGAN
from espnet2.gan_codec.shared.discriminator.msmpmb_discriminator import (
    MultiScaleMultiPeriodMultiBandDiscriminator,
)


class DummyEncoder(nn.Module):
    def __init__(self, dim=8, input_sr=16000):
        super().__init__()
        self.input_sr = input_sr
        self._dim = dim
        self.proj = nn.Linear(320, dim)

    @property
    def ssl_dim(self):
        return self._dim

    def _wav_to_ssl_inputs(self, wav, lengths):
        frames = wav.size(1) // 320
        x = wav[:, : frames * 320].reshape(wav.size(0), frames, 320)
        mask = (torch.arange(frames)[None] < (lengths // 320)[:, None]).long()
        return {"x": x, "attention_mask": mask}

    def encode(self, inputs, teacher=False):
        feat = self.proj(inputs["x"])
        return (feat.detach() if teacher else feat), inputs["attention_mask"]


def make_model(use_predicted_feat=False):
    discriminator = MultiScaleMultiPeriodMultiBandDiscriminator(
        rates=[],
        fft_sizes=[256],
        sample_rate=48000,
        periods=[2, 3],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 4,
            "downsample_scales": [3, 3, 1],
            "max_downsample_channels": 8,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
        band_discriminator_params={
            "hop_factor": 0.25,
            "sample_rate": 48000,
            "bands": [(0.0, 0.5), (0.5, 1.0)],
            "channel": 4,
        },
    )
    return SidonVocoderGAN(
        ssl_encoder=DummyEncoder(),
        vocoder=SidonVocoder(input_dim=8, channels=64, rates=[8, 5, 4, 3, 2]),
        discriminator=discriminator,
        use_predicted_feat=use_predicted_feat,
        segment_duration=0.1,
        mel_loss_conf={
            "fs": 48000,
            "n_fft": 512,
            "hop_length": 120,
            "win_length": 512,
            "n_mels": 32,
        },
    )


def test_upsample_factor_must_match_output_rate():
    with pytest.raises(ValueError):
        SidonVocoderGAN(
            ssl_encoder=DummyEncoder(),
            vocoder=SidonVocoder(input_dim=8, channels=64, rates=[2, 2]),
            discriminator=nn.Identity(),
        )


def test_generator_and_discriminator_turns():
    model = make_model().train()
    assert not model.ssl_encoder.training  # frozen encoder stays in eval
    speech = torch.randn(2, 24000) * 0.1  # 0.5 s at 48 kHz
    lengths = torch.tensor([24000, 19200])
    crop = torch.tensor([0, 3])
    gen = model(
        speech_ref1=speech,
        speech_ref1_lengths=lengths,
        vocoder_crop_start=crop,
        forward_generator=True,
    )
    assert gen["optim_idx"] == 0
    assert torch.isfinite(gen["loss"])
    gen["loss"].backward()
    assert all(p.grad is not None for p in model.vocoder.parameters())
    assert all(p.grad is None for p in model.ssl_encoder.parameters())

    disc = model(
        speech_ref1=speech,
        speech_ref1_lengths=lengths,
        vocoder_crop_start=crop,
        forward_generator=False,
    )
    assert disc["optim_idx"] == 1
    assert torch.isfinite(disc["loss"])
    disc["loss"].backward()
    assert all(p.grad is not None for p in model.discriminator.parameters())
    for key in ("loss_D", "loss_D_real", "loss_D_fake"):
        assert key in disc["stats"]


def test_crop_aligns_frames_and_samples():
    model = make_model()
    feat = torch.arange(2 * 25 * 8, dtype=torch.float32).reshape(2, 25, 8)
    wav = torch.arange(2 * 24000, dtype=torch.float32).reshape(2, 24000)
    feat_out, wav_out = model._crop(
        feat,
        torch.tensor([25, 10]),
        wav,
        torch.tensor([24000, 9600]),
        torch.tensor([3, 8]),
    )
    assert feat_out.shape == (2, 5, 8) and wav_out.shape == (2, 4800)
    torch.testing.assert_close(feat_out[0], feat[0, 3:8])
    torch.testing.assert_close(wav_out[0], wav[0, 3 * 960 : 8 * 960])
    # the second item has 10 frames, so the crop is clamped to start at 5
    torch.testing.assert_close(feat_out[1], feat[1, 5:10])


def test_predicted_features_need_noisy_speech():
    model = make_model(use_predicted_feat=True)
    speech = torch.randn(2, 24000) * 0.1
    lengths = torch.tensor([24000, 24000])
    with pytest.raises(ValueError):
        model(speech_ref1=speech, speech_ref1_lengths=lengths)
    out = model(
        speech_ref1=speech,
        speech_ref1_lengths=lengths,
        noisy_speech=torch.randn(2, 8000) * 0.1,
        noisy_speech_lengths=torch.tensor([8000, 8000]),
        forward_generator=True,
    )
    assert torch.isfinite(out["loss"])
