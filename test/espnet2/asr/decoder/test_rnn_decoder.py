import pytest
import torch

from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet.nets.beam_search import BeamSearch


@pytest.mark.parametrize("context_residual", [True, False])
@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_RNNDecoder_backward(context_residual, rnn_type):
    decoder = RNNDecoder(10, 12, context_residual=context_residual, rnn_type=rnn_type)
    x = torch.randn(2, 9, 12)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    t = torch.randint(0, 10, [2, 4], dtype=torch.long)
    t_lens = torch.tensor([4, 3], dtype=torch.long)
    z_all, ys_in_lens = decoder(x, x_lens, t, t_lens)
    z_all.sum().backward()


@pytest.mark.parametrize("context_residual", [True, False])
@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_RNNDecoder_init_state(context_residual, rnn_type):
    decoder = RNNDecoder(10, 12, context_residual=context_residual, rnn_type=rnn_type)
    x = torch.randn(9, 12)
    state = decoder.init_state(x)
    t = torch.randint(0, 10, [4], dtype=torch.long)
    decoder.score(t, state, x)


def test_RNNDecoder_invalid_type():
    with pytest.raises(ValueError):
        RNNDecoder(10, 12, rnn_type="foo")


@pytest.mark.parametrize("context_residual", [True, False])
@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_RNNDecoder_beam_search(context_residual, rnn_type, dtype):
    token_list = ["<blank>", "a", "b", "c", "unk", "<eos>"]
    vocab_size = len(token_list)
    encoder_output_size = 4
    decoder = RNNDecoder(
        vocab_size,
        encoder_output_size=encoder_output_size,
        context_residual=context_residual,
        rnn_type=rnn_type,
    )
    beam = BeamSearch(
        beam_size=3,
        vocab_size=vocab_size,
        weights={"test": 1.0},
        scorers={"test": decoder},
        token_list=token_list,
        sos=vocab_size - 1,
        eos=vocab_size - 1,
        pre_beam_score_key=None,
    )
    beam.to(dtype=dtype)

    enc = torch.randn(10, encoder_output_size).type(dtype)
    with torch.no_grad():
        beam(
            x=enc,
            maxlenratio=0.0,
            minlenratio=0.0,
        )
