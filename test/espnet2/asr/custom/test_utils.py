import pytest

from espnet2.asr.custom.build_encoder import build_encoder


def test_invalid_layer_type():
    architecture = [{"layer_type": "foo"}]

    with pytest.raises(NotImplementedError):
        _, _, _ = build_encoder(4, architecture)


def test_no_layer_type():
    architecture = [{"foo": "foo"}]

    with pytest.raises(ValueError):
        _, _, _ = build_encoder(4, architecture)


def test_invalid_layer_io():
    architecture = [
        {"layer_type": "embed", "hidden_size": 8},
        {"layer_type": "transformer", "hidden_size": 6},
    ]

    with pytest.raises(ValueError):
        _, _, _ = build_encoder(4, architecture)


def test_invalid_layer_parameters():
    architecture = [{"layer_type": "linear", "foo": "foo"}]

    with pytest.raises(ValueError):
        _, _, _ = build_encoder(4, architecture)


def test_invalid_positional_encoding_type():
    architecture = [
        {"layer_type": "linear", "hidden_size": 8},
        {"layer_type": "transformer", "hidden_size": 8},
    ]

    with pytest.raises(NotImplementedError):
        _, _, _ = build_encoder(4, architecture, positional_encoding_type="foo")


def test_invalid_positionwise_type():
    architecture = [
        {"layer_type": "linear", "hidden_size": 8},
        {"layer_type": "transformer", "hidden_size": 8},
    ]

    with pytest.raises(NotImplementedError):
        _, _, _ = build_encoder(4, architecture, positionwise_type="foo")


def test_invalid_self_attention_type():
    architecture = [
        {"layer_type": "linear", "hidden_size": 8},
        {"layer_type": "transformer", "hidden_size": 8},
    ]

    with pytest.raises(NotImplementedError):
        _, _, _ = build_encoder(4, architecture, self_attention_type="foo")
