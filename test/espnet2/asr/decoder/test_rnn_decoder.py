from typing import Tuple

from numpy import count_nonzero
import pytest
import torch

from espnet.nets.beam_search import BeamSearch
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet2.asr.decoder.rnn_decoder import RNNDecoder


@pytest.mark.parametrize("context_residual", [True, False])
@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_RNNDecoder_backward(context_residual, rnn_type, num_layers):
    decoder = RNNDecoder(
        10,
        12,
        num_layers=num_layers,
        context_residual=context_residual,
        rnn_type=rnn_type,
    )
    x = torch.randn(2, 9, 12)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    t = torch.randint(0, 10, [2, 4], dtype=torch.long)
    t_lens = torch.tensor([4, 3], dtype=torch.long)
    z_all, ys_in_lens = decoder(x, x_lens, t, t_lens)
    z_all.sum().backward()


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("use_attention", [False, True])
def test_transducer_decoder_backward(rnn_type, use_attention):
    decoder = RNNDecoder(
        10,
        12,
        num_layers=2,
        hidden_size=15,
        rnn_type=rnn_type,
        use_attention=use_attention,
        use_output=False,
    )
    joint_net = JointNetwork(10, 12, decoder.dunits, joint_space_size=20)

    x = torch.randn(2, 9, 12)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    t = torch.randint(0, 10, [2, 4], dtype=torch.long)
    t_lens = torch.tensor([4, 3], dtype=torch.long)

    decoder_out, _ = decoder(x, x_lens, t, t_lens)
    z_all = joint_net(x.unsqueeze(2), decoder_out.unsqueeze(1))

    z_all.sum().backward()


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("use_attention", [False, True])
def test_transducer_decoder_init_state(rnn_type, use_attention):
    decoder = RNNDecoder(
        10, 12, rnn_type=rnn_type, use_attention=use_attention, use_output=False
    )
    x = torch.randn(9, 12)
    state = decoder.init_state(x)

    if use_attention:
        assert isinstance(state[0], Tuple) and state[1] is None
        assert not count_nonzero(state[0][0][0]) and not count_nonzero(state[0][1][0])
    else:
        assert isinstance(state, Tuple)
        assert not count_nonzero(state[0][0]) and not count_nonzero(state[1][0])


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
