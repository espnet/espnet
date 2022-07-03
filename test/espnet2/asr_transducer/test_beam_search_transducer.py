import numpy as np
import pytest
import torch

from espnet2.asr_transducer.beam_search_transducer import (
    BeamSearchTransducer,
    Hypothesis,
)
from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr_transducer.decoder.stateless_decoder import StatelessDecoder
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.lm.seq_rnn_lm import SequentialRNNLM


@pytest.mark.parametrize(
    "decoder_class, decoder_opts, search_opts",
    [
        (
            RNNDecoder,
            {"hidden_size": 4},
            {"search_type": "default", "score_norm": True},
        ),
        (RNNDecoder, {"hidden_size": 4}, {"search_type": "default", "lm": None}),
        (StatelessDecoder, {}, {"search_type": "default", "lm": None}),
        (StatelessDecoder, {}, {"search_type": "default"}),
        (RNNDecoder, {"hidden_size": 4}, {"search_type": "alsd", "u_max": 10}),
        (
            RNNDecoder,
            {"hidden_size": 4},
            {"search_type": "alsd", "u_max": 10, "lm": None},
        ),
        (StatelessDecoder, {}, {"search_type": "alsd", "u_max": 10}),
        (StatelessDecoder, {}, {"search_type": "alsd", "u_max": 10, "lm": None}),
        (RNNDecoder, {"hidden_size": 4}, {"search_type": "tsd", "max_sym_exp": 3}),
        (
            RNNDecoder,
            {"hidden_size": 4},
            {"search_type": "tsd", "max_sym_exp": 3, "lm": None},
        ),
        (StatelessDecoder, {}, {"search_type": "tsd", "max_sym_exp": 3}),
        (StatelessDecoder, {}, {"search_type": "tsd", "max_sym_exp": 3, "lm": None}),
        (RNNDecoder, {"hidden_size": 4}, {"search_type": "maes", "nstep": 2}),
        (
            RNNDecoder,
            {"hidden_size": 4},
            {"search_type": "maes", "nstep": 2, "lm": None},
        ),
        (StatelessDecoder, {}, {"search_type": "maes", "nstep": 2}),
        (StatelessDecoder, {}, {"search_type": "maes", "nstep": 2, "lm": None}),
    ],
)
def test_transducer_beam_search(decoder_class, decoder_opts, search_opts):
    token_list = ["<blank>", "a", "b", "c"]
    vocab_size = len(token_list)

    encoder_size = 4

    decoder = decoder_class(vocab_size, embed_size=4, **decoder_opts)
    joint_net = JointNetwork(vocab_size, encoder_size, 4, joint_space_size=2)

    lm = search_opts.pop(
        "lm", SequentialRNNLM(vocab_size, unit=8, nlayers=1, rnn_type="lstm")
    )

    beam = BeamSearchTransducer(
        decoder,
        joint_net,
        beam_size=2,
        lm=lm,
        **search_opts,
    )

    enc_out = torch.randn(30, encoder_size)

    with torch.no_grad():
        _ = beam(enc_out)


@pytest.mark.parametrize(
    "search_opts",
    [
        {"beam_size": 5},
        {"beam_size": 2, "search_type": "tsd", "max_sym_exp": 1},
        {"beam_size": 2, "search_type": "alsd", "u_max": -2},
        {"beam_size": 2, "search_type": "maes", "expansion_beta": 2.3},
    ],
)
def test_integer_parameters_limits(search_opts):
    vocab_size = 4
    encoder_size = 4

    decoder = StatelessDecoder(vocab_size, embed_size=4)
    joint_net = JointNetwork(vocab_size, encoder_size, 4, joint_space_size=2)

    with pytest.raises(AssertionError):
        _ = BeamSearchTransducer(
            decoder,
            joint_net,
            **search_opts,
        )


def test_recombine_hyps():
    decoder = StatelessDecoder(4, embed_size=4)
    joint_net = JointNetwork(4, 4, 4, joint_space_size=2)
    beam_search = BeamSearchTransducer(decoder, joint_net, 2)

    test_hyp = [
        Hypothesis(score=0.0, yseq=[0, 1, 2], dec_state=None),
        Hypothesis(score=12.0, yseq=[0, 1, 2], dec_state=None),
    ]

    final = beam_search.recombine_hyps(test_hyp)

    assert len(final) == 1
    assert final[0].score == np.logaddexp(0.0, 12.0)
