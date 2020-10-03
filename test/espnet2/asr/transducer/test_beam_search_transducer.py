import pytest
import torch

from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("search_type", ["default", "alsd", "nsc", "tsd"])
@pytest.mark.parametrize("use_attention", [False, True])
@pytest.mark.parametrize("beam_size", [1, 2])
@pytest.mark.parametrize("nstep", [1, 2])
def test_transducer_decoder_beam_search(
    rnn_type, search_type, use_attention, beam_size, nstep
):
    encoder_output_size = 4

    decoder = RNNDecoder(
        10,
        encoder_output_size,
        hidden_size=12,
        embed_pad=0,
        rnn_type=rnn_type,
        num_layers=2,
        use_attention=use_attention,
        use_output=False,
    )
    joint_net = JointNetwork(
        10, encoder_output_size, decoder.dunits, joint_space_size=20
    )

    beam_search = BeamSearchTransducer(
        decoder=decoder,
        joint_network=joint_net,
        beam_size=beam_size,
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
