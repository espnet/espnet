from typing import Tuple

from numpy import count_nonzero
import pytest
import torch

from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("use_attention", [False, True])
def test_TransducerDecoder_backward(rnn_type, use_attention):
    decoder = TransducerDecoder(10, 12, rnn_type=rnn_type, use_attention=use_attention)
    x = torch.randn(2, 9, 12)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    t = torch.randint(0, 10, [2, 4], dtype=torch.long)

    z_all = decoder(x, t, x_lens)
    z_all.sum().backward()


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("use_attention", [False, True])
def test_TransducerDecoder_init_state(rnn_type, use_attention):
    decoder = TransducerDecoder(10, 12, rnn_type=rnn_type, use_attention=use_attention)
    x = torch.randn(9, 12)
    state = decoder.init_state(x)

    if use_attention:
        assert isinstance(state[0], Tuple) and state[1] is None
        assert not count_nonzero(state[0][0][0]) and not count_nonzero(state[0][1][0])
    else:
        assert isinstance(state, Tuple)
        assert not count_nonzero(state[0][0]) and not count_nonzero(state[1][0])


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("search_type", ["default", "alsd", "nsc", "tsd"])
@pytest.mark.parametrize("use_attention", [False, True])
@pytest.mark.parametrize("nstep", [1, 2])
def test_TransducerDecoder_beam_search(rnn_type, search_type, use_attention, nstep):
    encoder_output_size = 4

    decoder = TransducerDecoder(
        10, encoder_output_size, rnn_type=rnn_type, use_attention=use_attention
    )

    beam_search = BeamSearchTransducer(
        decoder=decoder,
        beam_size=2,
        lm=None,
        lm_weight=0.0,
        search_type=search_type,
        max_sym_exp=2,
        u_max=10,
        nstep=nstep,
        prefix_alpha=2,
        score_norm=False,
    )

    enc = torch.randn(10, encoder_output_size)
    with torch.no_grad():
        beam_search(enc)
