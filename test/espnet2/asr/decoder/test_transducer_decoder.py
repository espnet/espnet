import pytest
import torch

from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.transducer.beam_search_transducer import Hypothesis


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_TransducerDecoder_forward(rnn_type):
    ys = torch.randint(0, 10, [4, 10], dtype=torch.long)
    decoder = TransducerDecoder(10, rnn_type=rnn_type)

    decoder.set_device(ys.device)
    _ = decoder(ys)


def test_TransducerDecoder_invalid_type():
    with pytest.raises(ValueError):
        TransducerDecoder(10, rnn_type="foo")


def test_TransducerDecoder_score():
    decoder = TransducerDecoder(10, rnn_type="lstm")
    dec_state = decoder.init_state(1)
    hyp = Hypothesis(score=0.0, yseq=[0], dec_state=dec_state)

    _, _, _ = decoder.score(hyp, {})


def test_TransducerDecoder_batch_score():
    decoder = TransducerDecoder(10, rnn_type="lstm")
    batch_state = decoder.init_state(3)
    hyps = [
        Hypothesis(score=0.0, yseq=[0], dec_state=decoder.select_state(batch_state, 0))
    ]

    _, _, _ = decoder.batch_score(hyps, batch_state, {}, True)


def test_TransducerDecoder_cache_score():
    decoder = TransducerDecoder(10, rnn_type="gru")

    batch_state = decoder.init_state(3)

    hyps = [
        Hypothesis(score=0.0, yseq=[0], dec_state=decoder.select_state(batch_state, 0))
    ]

    cache = {"0": hyps[0].dec_state}
    dec_out, _, _ = decoder.score(hyps[0], cache)

    batch_cache = {"0": (dec_out.view(1, 1, -1), hyps[0].dec_state)}
    _, _, _ = decoder.batch_score(hyps, batch_state, batch_cache, False)
