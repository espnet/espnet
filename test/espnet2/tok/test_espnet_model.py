from test.espnet2.tok.test_hifigan_loss import make_loss

import pytest
import torch

from espnet2.tok.espnet_model import ESPnetGANSpeechTokenizerModel
from espnet2.tok.output import SpeechTokenizerOutput


class DummyTokenizer(torch.nn.Module):
    def __init__(self, logits=(0.2, 0.8)):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits))
        self.forward_calls = 0
        self.encode_calls = 0

    def _output(self, speech, speech_lengths):
        batch, time = speech.shape
        assignment = torch.softmax(self.logits, dim=0).expand(batch, time, -1)
        return SpeechTokenizerOutput(
            continuous=speech.unsqueeze(-1),
            assignment=assignment,
            token_ids=assignment.argmax(dim=-1),
            lengths=speech_lengths,
        )

    def forward(self, speech, speech_lengths):
        self.forward_calls += 1
        return self._output(speech, speech_lengths)

    def encode(self, speech, speech_lengths):
        self.encode_calls += 1
        return self._output(speech, speech_lengths)


class DummyASRLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_loss = None

    def forward(self, tokenizer_output, **batch):
        loss = tokenizer_output.assignment[..., 0].mean()
        self.last_loss = loss.detach()
        return loss, {"loss": loss.detach()}, tokenizer_output.assignment.size(0)

    def encode(self, tokenizer_output):
        return tokenizer_output.assignment, tokenizer_output.lengths


class DummyReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = torch.nn.Linear(1, 1)
        self.has_cached_generator_outputs = False
        self.last_discriminator_tokenizer_output = None
        self.last_loss = None

    def forward(self, tokenizer_output, **batch):
        self.has_cached_generator_outputs = True
        loss = tokenizer_output.assignment[..., 1].mean()
        self.last_loss = loss.detach()
        return loss, {"loss": loss.detach()}, tokenizer_output.assignment.size(0)

    def forward_discriminator(self, tokenizer_output, speech, **batch):
        self.last_discriminator_tokenizer_output = tokenizer_output
        self.has_cached_generator_outputs = False
        loss = self.discriminator(speech.unsqueeze(-1)).square().mean()
        return loss, {"loss": loss.detach()}, speech.size(0)

    def synthesize(self, tokenizer_output, spembs):
        return tokenizer_output.assignment[:, :1, :1], tokenizer_output.lengths


def make_model(reconstruction_weight=0.25):
    return ESPnetGANSpeechTokenizerModel(
        tokenizer=DummyTokenizer(),
        asr_loss=DummyASRLoss(),
        reconstruction_loss=DummyReconstructionLoss(),
        reconstruction_weight=reconstruction_weight,
    )


def test_generator_combines_losses_and_tokenizes_once():
    model = make_model()
    result = model(
        speech=torch.randn(2, 4),
        speech_lengths=torch.tensor([4, 3]),
        text=torch.ones(2, 2, dtype=torch.long),
        text_lengths=torch.tensor([2, 1]),
        spembs=torch.randn(2, 3),
        forward_generator=True,
    )

    expected = (
        0.75 * model.asr_loss.last_loss + 0.25 * model.reconstruction_loss.last_loss
    )
    torch.testing.assert_close(result["loss"], expected[None])
    assert result["optim_idx"] == 0
    # ASR and reconstruction must share one tokenizer forward pass.
    assert model.tokenizer.forward_calls == 1
    assert "asr_loss" in result["stats"]
    assert "reconstruction_loss" in result["stats"]

    result["loss"].backward()
    assert model.tokenizer.logits.grad is not None


def test_generator_skips_inactive_asr_loss_without_text():
    model = make_model(reconstruction_weight=1.0)
    result = model(
        speech=torch.randn(2, 4),
        speech_lengths=torch.tensor([4, 3]),
        spembs=torch.randn(2, 3),
        forward_generator=True,
    )

    assert model.asr_loss.last_loss is None
    assert model.reconstruction_loss.last_loss is not None
    assert "asr_loss" not in result["stats"]
    assert "reconstruction_loss" in result["stats"]


def test_generator_skips_inactive_reconstruction_loss():
    model = make_model(reconstruction_weight=0.0)
    result = model(
        speech=torch.randn(2, 4),
        speech_lengths=torch.tensor([4, 3]),
        text=torch.ones(2, 2, dtype=torch.long),
        text_lengths=torch.tensor([2, 1]),
        spembs=torch.randn(2, 3),
        forward_generator=True,
    )

    assert model.asr_loss.last_loss is not None
    assert model.reconstruction_loss.last_loss is None
    assert "asr_loss" in result["stats"]
    assert "reconstruction_loss" not in result["stats"]


def test_discriminator_uses_cache_without_running_tokenizer():
    model = make_model()
    inputs = {
        "speech": torch.randn(2, 4),
        "speech_lengths": torch.tensor([4, 3]),
        "text": torch.ones(2, 2, dtype=torch.long),
        "text_lengths": torch.tensor([2, 1]),
        "spembs": torch.randn(2, 3),
    }
    model(forward_generator=True, **inputs)
    result = model(forward_generator=False, **inputs)

    assert result["optim_idx"] == 1
    # The generator already ran the tokenizer, so the cache avoids another call.
    assert model.tokenizer.forward_calls == 1
    # Cached waveforms let the discriminator run without tokenizer outputs.
    assert model.reconstruction_loss.last_discriminator_tokenizer_output is None
    assert "reconstruction_discriminator_loss" in result["stats"]


def test_discriminator_tokenizes_on_cache_miss():
    model = make_model()
    result = model(
        speech=torch.randn(2, 4),
        speech_lengths=torch.tensor([4, 3]),
        spembs=torch.randn(2, 3),
        forward_generator=False,
    )

    assert result["optim_idx"] == 1
    # A cache miss requires tokenization for waveform generation.
    assert model.tokenizer.forward_calls == 1
    # The freshly computed tokenizer output is passed to the discriminator path.
    assert model.reconstruction_loss.last_discriminator_tokenizer_output is not None


def make_model_with_hifigan_loss():
    # make_loss() builds a 4-token, 2-dim-speaker tiny HiFi-GAN with caching.
    return ESPnetGANSpeechTokenizerModel(
        tokenizer=DummyTokenizer(logits=(0.1, 0.2, 0.3, 0.4)),
        asr_loss=DummyASRLoss(),
        reconstruction_loss=make_loss(),
        reconstruction_weight=0.5,
    )


def make_batch():
    return {
        "speech": torch.randn(2, 8),
        "speech_lengths": torch.tensor([8, 6]),
        "text": torch.ones(2, 2, dtype=torch.long),
        "text_lengths": torch.tensor([2, 1]),
        "spembs": torch.randn(2, 2),
    }


def test_generator_first_reuses_cache_across_batches():
    model = make_model_with_hifigan_loss()
    model.train()
    for _ in range(2):
        batch = make_batch()
        model(forward_generator=True, **batch)
        assert model.reconstruction_loss.has_cached_generator_outputs
        model(forward_generator=False, **batch)
        assert not model.reconstruction_loss.has_cached_generator_outputs

    # The discriminator step reused the cache instead of tokenizing again.
    assert model.tokenizer.forward_calls == 2


def test_discriminator_first_fails_on_stale_cache():
    model = make_model_with_hifigan_loss()
    model.train()
    first, second = make_batch(), make_batch()
    model(forward_generator=False, **first)
    model(forward_generator=True, **first)

    with pytest.raises(RuntimeError, match="different speech tensor"):
        model(forward_generator=False, **second)


@pytest.mark.parametrize("reconstruction_weight", [-0.1, 1.1])
def test_reconstruction_weight_must_be_between_zero_and_one(
    reconstruction_weight,
):
    with pytest.raises(ValueError, match="reconstruction_weight"):
        ESPnetGANSpeechTokenizerModel(
            tokenizer=DummyTokenizer(),
            asr_loss=DummyASRLoss(),
            reconstruction_loss=DummyReconstructionLoss(),
            reconstruction_weight=reconstruction_weight,
        )


def test_inference_interfaces_use_deterministic_tokenizer():
    model = make_model()
    speech = torch.randn(1, 3)
    lengths = torch.tensor([3])

    tokenizer_output = model.tokenize(speech, lengths)
    encoded, encoded_lengths = model.encode_asr(speech, lengths)
    waveform, waveform_lengths = model.synthesize(
        speech, lengths, spembs=torch.randn(1, 3)
    )

    assert tokenizer_output.token_ids.shape == (1, 3)
    assert encoded.shape == (1, 3, 2)
    torch.testing.assert_close(encoded_lengths, lengths)
    assert waveform.shape == (1, 1, 1)
    torch.testing.assert_close(waveform_lengths, lengths)
    # Each public inference interface independently calls tokenizer.encode().
    assert model.tokenizer.encode_calls == 3
