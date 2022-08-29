import pytest
import torch

from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch


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


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "RNN_TANH", "RNN_RELU"])
@pytest.mark.parametrize("tie_weights", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_SequentialRNNLM_beam_search(rnn_type, tie_weights, dtype):
    token_list = ["<blank>", "a", "b", "c", "unk", "<eos>"]
    vocab_size = len(token_list)

    model = SequentialRNNLM(
        vocab_size, nlayers=2, rnn_type=rnn_type, tie_weights=tie_weights
    )
    beam = BeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"test": 1.0},
        scorers={"test": model},
        token_list=token_list,
        sos=vocab_size - 1,
        eos=vocab_size - 1,
        pre_beam_score_key=None,
    )
    beam.to(dtype=dtype)

    enc = torch.randn(10, 20).type(dtype)
    with torch.no_grad():
        beam(
            x=enc, maxlenratio=0.0, minlenratio=0.0,
        )


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "RNN_TANH", "RNN_RELU"])
@pytest.mark.parametrize("tie_weights", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_SequentialRNNLM_batch_beam_search(rnn_type, tie_weights, dtype):
    token_list = ["<blank>", "a", "b", "c", "unk", "<eos>"]
    vocab_size = len(token_list)

    model = SequentialRNNLM(
        vocab_size, nlayers=2, rnn_type=rnn_type, tie_weights=tie_weights
    )
    beam = BatchBeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"test": 1.0},
        scorers={"test": model},
        token_list=token_list,
        sos=vocab_size - 1,
        eos=vocab_size - 1,
        pre_beam_score_key=None,
    )
    beam.to(dtype=dtype)

    enc = torch.randn(10, 20).type(dtype)
    with torch.no_grad():
        beam(
            x=enc, maxlenratio=0.0, minlenratio=0.0,
        )
