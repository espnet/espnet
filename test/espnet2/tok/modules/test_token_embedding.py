import pytest
import torch

from espnet2.tok.modules.token_embedding import TokenEmbedding


def test_token_ids_and_one_hot_have_identical_outputs():
    module = TokenEmbedding(4, 3, use_positional_encoding=False)
    token_ids = torch.tensor([[0, 2], [1, 3]])
    lengths = torch.tensor([2, 1])
    one_hot = torch.nn.functional.one_hot(token_ids, num_classes=4).float()

    id_output, id_lengths = module(token_ids, lengths)
    distribution_output, distribution_lengths = module(one_hot, lengths)

    torch.testing.assert_close(id_output, distribution_output)
    assert id_lengths is lengths
    assert distribution_lengths is lengths
    assert module.output_size() == 3


def test_distribution_path_propagates_gradient_to_assignments():
    module = TokenEmbedding(3, 2, use_positional_encoding=False)
    assignments = torch.tensor([[[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]]], requires_grad=True)

    output, _ = module(assignments, torch.tensor([2]))
    output.square().sum().backward()

    assert assignments.grad is not None
    assert torch.count_nonzero(assignments.grad) > 0
    assert module.embed.weight.grad is not None


@pytest.mark.parametrize(
    "input, lengths, message",
    [
        (torch.zeros(2, 4), torch.tensor([4, 4]), "Floating-point input"),
        (torch.zeros(2, 4, 5), torch.tensor([4, 4]), "distribution size"),
        (torch.zeros(2, 4, 3, dtype=torch.long), torch.tensor([4, 4]), "Integer input"),
        (
            torch.zeros(2, 4, dtype=torch.long),
            torch.tensor([[4], [4]]),
            "input_lengths",
        ),
        (torch.zeros(2, 4, dtype=torch.long), torch.tensor([4]), "Batch sizes"),
    ],
)
def test_invalid_shapes(input, lengths, message):
    module = TokenEmbedding(4, 3, use_positional_encoding=False)
    with pytest.raises(ValueError, match=message):
        module(input, lengths)
