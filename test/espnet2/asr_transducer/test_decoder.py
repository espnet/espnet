import pytest
import torch

from espnet2.asr_transducer.decoder.mega_decoder import MEGADecoder
from espnet2.asr_transducer.decoder.rnn_decoder import RNNDecoder
from espnet2.asr_transducer.decoder.rwkv_decoder import RWKVDecoder
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU is required for WKV kernel computation.",
)
@pytest.mark.parametrize(
    "params",
    [
        {"block_size": 4, "num_blocks": 2},
        {"block_size": 4, "num_blocks": 2, "attention_size": 8, "linear_size": 8},
    ],
)
@pytest.mark.execution_timeout(20)
def test_rwkv_decoder(params):
    vocab_size, labels = prepare()

    decoder = RWKVDecoder(vocab_size, **params)
    _ = decoder(labels)


def test_stateless_decoder():
    vocab_size, labels = prepare()

    decoder = StatelessDecoder(vocab_size, embed_size=2)
    _ = decoder(labels)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"rel_pos_bias_type": "rotary"},
        {"chunk_size": 8},
        {"chunk_size": 16},
    ],
)
def test_mega_decoder(params):
    vocab_size, labels = prepare()

    decoder = MEGADecoder(vocab_size, **params)
    _ = decoder(labels)


def test_mega_rel_pos_bias_type():
    vocab_size, _ = prepare()

    with pytest.raises(ValueError):
        _ = MEGADecoder(vocab_size, rel_pos_bias_type="foo")


@pytest.mark.parametrize(
    "rel_pos_bias_type",
    ["simple", "rotary"],
)
def test_mega_rel_pos_bias(rel_pos_bias_type):
    vocab_size, labels = prepare()

    decoder = MEGADecoder(
        vocab_size, max_positions=1, rel_pos_bias_type=rel_pos_bias_type
    )

    if rel_pos_bias_type == "simple":
        with pytest.raises(ValueError):
            _ = decoder(labels)
    else:
        _ = decoder(labels)


def test_rnn_type():
    vocab_size, _ = prepare()

    with pytest.raises(ValueError):
        _ = RNNDecoder(vocab_size, rnn_type="foo")
