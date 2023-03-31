import pytest
import torch

from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize(
    "search_params",
    [
        {"search_type": "greedy"},
        {"search_type": "default", "score_norm": False, "nbest": 4},
        {
            "search_type": "default",
            "score_norm": False,
            "nbest": 4,
            "lm": "TransformerLM",
        },
        {"search_type": "alsd", "u_max": 20},
        {"search_type": "tsd", "max_sym_exp": 3},
        {"search_type": "nsc", "nstep": 2, "lm": None},
        {"search_type": "nsc", "nstep": 2},
        {"search_type": "maes", "nstep": 2, "lm": None},
        {"search_type": "maes", "nstep": 2},
        {
            "search_type": "mbg",
            "multi_blank_durations": [2, 1],
            "multi_blank_indices": [0, 1],
        },
    ],
)
def test_transducer_beam_search(rnn_type, search_params):
    token_list = ["<blank>", "a", "b", "c", "<sos>"]
    vocab_size = len(token_list)
    beam_size = 1 if search_params["search_type"] in ["greedy", "mbg"] else 2

    encoder_output_size = 4
    decoder_output_size = 4

    decoder = TransducerDecoder(
        vocab_size, hidden_size=decoder_output_size, rnn_type=rnn_type
    )
    joint_net = JointNetwork(
        vocab_size, encoder_output_size, decoder_output_size, joint_space_size=2
    )

    lm = search_params.pop("lm", SequentialRNNLM(vocab_size, rnn_type="lstm"))
    if isinstance(lm, str) and lm == "TransformerLM":
        lm = TransformerLM(vocab_size, pos_enc=None, unit=10, layer=2)

    beam = BeamSearchTransducer(
        decoder,
        joint_net,
        beam_size=beam_size,
        lm=lm,
        token_list=token_list,
        **search_params,
    )

    enc_out = torch.randn(30, encoder_output_size)

    with torch.no_grad():
        _ = beam(enc_out)
