import pytest
import torch

from espnet2.lm.seq_rnn import SequentialRNNLM


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "RNN_TANH", "RNN_RELU"])
@pytest.mark.parametrize("tie_weights", [True, False])
def test_SequentialRNNLM_backward(rnn_type, tie_weights):
    model = SequentialRNNLM(10, rnn_type=rnn_type, tie_weights=tie_weights)
    input = torch.randint(0, 9, [2, 10])

    out, h = model(input, None)
    out, h = model(input, h)
    out.sum().backward()


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "RNN_TANH", "RNN_RELU"])
@pytest.mark.parametrize("tie_weights", [True, False])
def test_SequentialRNNLM_score(rnn_type, tie_weights):
    model = SequentialRNNLM(10, rnn_type=rnn_type, tie_weights=tie_weights)
    input = torch.randint(0, 9, (12,))
    state = model.init_state(None)
    model.score(input, state, None)


def test_SequentialRNNLM_invalid_type():
    with pytest.raises(ValueError):
        SequentialRNNLM(10, rnn_type="foooo")


def test_SequentialRNNLM_tie_weights_value_error():
    with pytest.raises(ValueError):
        SequentialRNNLM(10, tie_weights=True, unit=20, nhid=10)
