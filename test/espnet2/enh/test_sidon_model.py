from types import SimpleNamespace

import torch
from torch import nn

from espnet2.enh.sidon_model import SSL_ENCODERS, SidonFeaturePredictor, W2VBert2Encoder


class DummyEncoder(nn.Module):
    """Stands in for the SSL backbones: 50 Hz features from 16 kHz audio."""

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


def test_registry_names():
    assert set(SSL_ENCODERS) == {"w2v_bert2", "xeus"}


def test_wav_to_ssl_inputs_shapes_and_short_input():
    encoder = SimpleNamespace(input_sr=16000)
    wav = torch.randn(2, 16000) * 0.1
    lengths = torch.tensor([16000, 4000])
    inputs = W2VBert2Encoder._wav_to_ssl_inputs(encoder, wav, lengths)
    feats, mask = inputs["input_features"], inputs["attention_mask"]
    assert feats.dim() == 3 and feats.size(-1) == 160
    assert mask.shape == feats.shape[:2]
    assert torch.isfinite(feats).all()
    assert mask[0].sum() > mask[1].sum() > 0
    # 400 samples is a single fbank frame: the normalisation must not produce
    # NaN from an unbiased variance over one frame
    short = W2VBert2Encoder._wav_to_ssl_inputs(
        encoder, wav[:, :400], torch.tensor([400, 400])
    )
    assert torch.isfinite(short["input_features"]).all()


def test_feature_predictor_masked_loss():
    model = SidonFeaturePredictor(DummyEncoder())
    noisy = torch.randn(2, 16000)
    clean = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 8000])
    loss, stats, weight = model(
        noisy_speech=noisy,
        noisy_speech_lengths=lengths,
        speech_ref1=clean,
        speech_ref1_lengths=lengths,
    )
    assert torch.isfinite(loss)
    assert int(weight) == 2
    assert "loss" in stats
    loss.backward()
    assert model.ssl_encoder.proj.weight.grad is not None

    # padded frames do not change the loss: pad the shorter item further
    noisy2 = torch.cat([noisy, torch.zeros(2, 3200)], dim=1)
    clean2 = torch.cat([clean, torch.zeros(2, 3200)], dim=1)
    loss2, _, _ = model(
        noisy_speech=noisy2,
        noisy_speech_lengths=lengths,
        speech_ref1=clean2,
        speech_ref1_lengths=lengths,
    )
    torch.testing.assert_close(loss2, loss, atol=1e-6, rtol=1e-5)
