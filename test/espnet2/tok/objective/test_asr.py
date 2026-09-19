import pytest
import torch

from espnet2.tok.modules.token_embedding import TokenEmbedding
from espnet2.tok.objective.asr import ASRObjective
from espnet2.tok.output import SpeechTokenizerOutput


class DummyASRModel(torch.nn.Module):
    def __init__(self, frontend=None):
        super().__init__()
        self.frontend = frontend
        self.projection = torch.nn.Linear(3, 1)
        self.last_input_features = None

    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        self.last_input_features = speech
        loss = self.projection(speech).square().mean()
        return loss, {"loss": loss.detach()}, speech.size(0)

    def encode(self, speech, speech_lengths):
        self.last_input_features = speech
        return self.projection(speech), speech_lengths


def make_tokenizer_output(assignment):
    batch_size, time, _ = assignment.shape
    return SpeechTokenizerOutput(
        continuous=torch.zeros(batch_size, time, 2),
        assignment=assignment,
        token_ids=assignment.argmax(dim=-1),
        lengths=torch.tensor([time] * batch_size),
    )


def test_forward_embeds_assignments_and_preserves_gradient():
    asr_model = DummyASRModel()
    embedding = TokenEmbedding(4, 3, use_positional_encoding=False)
    objective = ASRObjective(asr_model, embedding)
    assignment = torch.nn.functional.one_hot(
        torch.tensor([[0, 2], [1, 3]]), num_classes=4
    ).float()
    assignment.requires_grad_()

    loss, stats, weight = objective(
        make_tokenizer_output(assignment),
        text=torch.tensor([[1], [2]]),
        text_lengths=torch.tensor([1, 1]),
    )
    loss.backward()

    # Forward uses the hard assignment, while backward still reaches it.
    expected = torch.matmul(assignment.detach(), embedding.embed.weight.detach())
    torch.testing.assert_close(asr_model.last_input_features.detach(), expected)
    assert assignment.grad is not None
    assert torch.count_nonzero(assignment.grad) > 0
    assert "loss" in stats
    assert weight == 2


def test_encode_uses_the_same_embedding():
    asr_model = DummyASRModel()
    embedding = TokenEmbedding(4, 3, use_positional_encoding=False)
    objective = ASRObjective(asr_model, embedding)
    assignment = torch.nn.functional.one_hot(
        torch.tensor([[0, 2]]), num_classes=4
    ).float()
    tokenizer_output = make_tokenizer_output(assignment)

    encoded, lengths = objective.encode(tokenizer_output)

    assert encoded.shape == (1, 2, 1)
    torch.testing.assert_close(lengths, tokenizer_output.lengths)


def test_rejects_asr_model_with_an_existing_frontend():
    with pytest.raises(ValueError, match="frontend=None"):
        ASRObjective(
            DummyASRModel(frontend=torch.nn.Identity()),
            TokenEmbedding(4, 4),
        )
