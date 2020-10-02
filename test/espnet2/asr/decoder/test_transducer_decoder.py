from typing import Tuple

from numpy import count_nonzero
import pytest
import torch

from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("use_attention", [False, True])
def test_transducer_decoder_backward(rnn_type, use_attention):
    decoder = RNNDecoder(
        10,
        12,
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


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("search_type", ["default", "alsd", "nsc", "tsd"])
@pytest.mark.parametrize("use_attention", [False, True])
@pytest.mark.parametrize("nstep", [1, 2])
def test_transducer_decoder_beam_search(rnn_type, search_type, use_attention, nstep):
    encoder_output_size = 4

    decoder = RNNDecoder(
        10,
        encoder_output_size,
        hidden_size=12,
        embed_pad=0,
        rnn_type=rnn_type,
        use_attention=use_attention,
        use_output=False,
    )
    joint_net = JointNetwork(
        10, encoder_output_size, decoder.dunits, joint_space_size=20
    )

    beam_search = BeamSearchTransducer(
        decoder=decoder,
        joint_network=joint_net,
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
