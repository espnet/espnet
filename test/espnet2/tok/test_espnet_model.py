import pytest
import torch

from espnet2.tok.espnet_model import ESPnetGANSpeechTokenizerModel
from espnet2.tok.output import SpeechTokenizerOutput


class DummyTokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, 0.8]))
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


class DummyASRObjective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_loss = None

    def forward(self, tokenizer_output, **batch):
        loss = tokenizer_output.assignment[..., 0].mean()
        self.last_loss = loss.detach()
        return loss, {"loss": loss.detach()}, tokenizer_output.assignment.size(0)

    def encode(self, tokenizer_output):
        return tokenizer_output.assignment, tokenizer_output.lengths


class DummyReconstructionObjective(torch.nn.Module):
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


def make_model():
    return ESPnetGANSpeechTokenizerModel(
        tokenizer=DummyTokenizer(),
        asr_objective=DummyASRObjective(),
        reconstruction_objective=DummyReconstructionObjective(),
        reconstruction_weight=0.25,
    )


def test_generator_combines_objectives_and_tokenizes_once():
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
        0.75 * model.asr_objective.last_loss
        + 0.25 * model.reconstruction_objective.last_loss
    )
    torch.testing.assert_close(result["loss"], expected[None])
    assert result["optim_idx"] == 0
    # ASR and reconstruction must share one tokenizer forward pass.
    assert model.tokenizer.forward_calls == 1
    assert "asr_loss" in result["stats"]
    assert "reconstruction_loss" in result["stats"]

    result["loss"].backward()
    assert model.tokenizer.logits.grad is not None


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
    assert model.reconstruction_objective.last_discriminator_tokenizer_output is None
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
    assert (
        model.reconstruction_objective.last_discriminator_tokenizer_output is not None
    )


@pytest.mark.parametrize("reconstruction_weight", [-0.1, 1.1])
def test_reconstruction_weight_must_be_between_zero_and_one(
    reconstruction_weight,
):
    with pytest.raises(ValueError, match="reconstruction_weight"):
        ESPnetGANSpeechTokenizerModel(
            tokenizer=DummyTokenizer(),
            asr_objective=DummyASRObjective(),
            reconstruction_objective=DummyReconstructionObjective(),
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
