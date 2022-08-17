import pytest
import torch

from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr_transducer.decoder.stateless_decoder import StatelessDecoder


def prepare():
    batch_size = 2
    vocab_size = 4
    n_token = vocab_size - 1

    label_len = [13, 9]
    labels = (torch.rand(batch_size, max(label_len)) * n_token % n_token).long()
    for i in range(2):
        labels[i, label_len[i] :] = 0

    return vocab_size, labels


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"embed_size": 2, "hidden_size": 4, "rnn_type": "gru", "num_layers": 2},
        {"rnn_type": "lstm", "num_layers": 2, "dropout_rate": 0.1},
    ],
)
def test_rnn_decoder(params):
    vocab_size, labels = prepare()

    decoder = RNNDecoder(vocab_size, **params)
    _ = decoder(labels)


def test_stateless_decoder():
    vocab_size, labels = prepare()

    decoder = StatelessDecoder(vocab_size, embed_size=2)
    _ = decoder(labels)


def test_rnn_type():
    vocab_size, labels = prepare()

    with pytest.raises(ValueError):
        _ = RNNDecoder(vocab_size, rnn_type="foo")
