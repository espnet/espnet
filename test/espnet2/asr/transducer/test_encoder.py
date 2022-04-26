import pytest
import torch

from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.utils import TooShortUttError


@pytest.mark.parametrize(
    "input_conf, body_conf, main_conf",
    [
        ({}, [{"block_type": "rnn", "dim_hidden": 4, "bidirectional": False}], {}),
        ({}, [{"block_type": "rnn", "dim_hidden": 4, "dim_proj": 2}], {}),
        (
            {},
            [
                {
                    "block_type": "rnn",
                    "dim_hidden": 4,
                    "dim_proj": 2,
                    "dim_output": 2,
                    "num_blocks": 2,
                    "subsample": (1, 2),
                }
            ],
            {},
        ),
        (
            {},
            [
                {
                    "block_type": "conformer",
                    "dim_hidden": 4,
                    "dim_linear": 2,
                    "macaron_style": True,
                    "conv_mod_kernel": 1,
                    "num_blocks": 2,
                }
            ],
            {"pos_enc_layer_type": "rel_pos"},
        ),
        (
            {},
            [
                {
                    "block_type": "conformer",
                    "dim_hidden": 4,
                    "dim_linear": 2,
                    "avg_eps": 1e-8,
                }
            ],
            {},
        ),
        (
            {},
            [
                {
                    "block_type": "conv1d",
                    "dim_output": 4,
                    "kernel_size": 1,
                    "use_batchnorm": True,
                }
            ],
            {},
        ),
        (
            {},
            [
                {"block_type": "conv1d", "dim_output": 4, "kernel_size": 1},
                {"block_type": "rnn", "dim_hidden": 4},
            ],
            {},
        ),
        (
            {},
            [
                {"block_type": "conv1d", "dim_output": 4, "kernel_size": 1},
                {"block_type": "rnn", "dim_hidden": 4},
                {"block_type": "conv1d", "dim_output": 2, "kernel_size": 1},
            ],
            {},
        ),
        (
            {},
            [
                {
                    "block_type": "conv1d",
                    "dim_output": 4,
                    "kernel_size": 2,
                    "dilation": 2,
                },
                {
                    "block_type": "conformer",
                    "dim_hidden": 4,
                    "dim_linear": 2,
                    "num_blocks": 2,
                },
                {"block_type": "conv1d", "dim_output": 4, "kernel_size": 1},
            ],
            {"pos_enc_layer_type": "rel_pos"},
        ),
        (
            {},
            [
                {"block_type": "conv1d", "dim_output": 4, "kernel_size": 1},
                {"block_type": "rnn", "dim_hidden": 4},
                {"block_type": "conv1d", "dim_output": 8, "kernel_size": 2},
            ],
            {},
        ),
        (
            {},
            [
                {"block_type": "rnn", "dim_hidden": 4, "num_blocks": 2},
                {"block_type": "conv1d", "dim_output": 4, "kernel_size": 1},
                {"block_type": "rnn", "dim_hidden": 4},
            ],
            {},
        ),
        (
            {"block_type": "conv2d", "subsampling_factor": 2},
            [{"block_type": "rnn", "dim_hidden": 4}],
            {},
        ),
        (
            {"block_type": "conv2d"},
            [{"block_type": "conv1d", "dim_output": 4, "kernel_size": 1}],
            {},
        ),
        (
            {"block_type": "conv2d", "dim_conv": 8},
            [{"block_type": "conv1d", "dim_output": 4, "kernel_size": 1}],
            {},
        ),
        (
            {"block_type": "conv2d", "dim_conv": 4},
            [{"block_type": "conformer", "dim_hidden": 4, "dim_linear": 2}],
            {},
        ),
        (
            {"block_type": "conv2d", "dim_conv": 4},
            [{"block_type": "conformer", "dim_hidden": 4, "dim_linear": 2}],
            {"pos_enc_layer_type": "rel_pos"},
        ),
        (
            {"block_type": "conv2d", "dim_conv": 4},
            [
                {
                    "block_type": "conv1d",
                    "dim_output": 4,
                    "kernel_size": 2,
                    "dilation": 2,
                },
                {
                    "block_type": "conformer",
                    "dim_hidden": 4,
                    "dim_linear": 2,
                    "num_blocks": 2,
                },
            ],
            {"pos_enc_layer_type": "rel_pos"},
        ),
    ],
)
def test_encoder(input_conf, body_conf, main_conf):
    input_size = 8

    encoder = Encoder(input_size, body_conf, input_conf=input_conf, main_conf=main_conf)

    sequence = torch.randn(2, 30, 8, requires_grad=True)
    sequence_len = torch.tensor([30, 18], dtype=torch.long)

    _ = encoder(sequence, sequence_len)


@pytest.mark.parametrize(
    "input_conf, body_conf",
    [
        ({}, [{}]),
        ({}, [{"block_type": "foo"}]),
        ({"block_type": "foo"}, [{"block_type": "rnn", "dim_hidden": 4}]),
    ],
)
def test_block_type(input_conf, body_conf):
    with pytest.raises(ValueError):
        _ = Encoder(8, body_conf, input_conf=input_conf)


@pytest.mark.parametrize(
    "body_conf",
    [
        [{"block_type": "conformer", "dim_hidden": 4}],
        [{"block_type": "conv1d"}],
        [
            {"block_type": "rnn", "dim_hidden": 4},
            {"block_type": "conformer", "dim_hidden": 4, "dim_linear": 2},
        ],
        [{"block_type": "rnn", "dim_hidden": 4, "rnn_type": "foo"}],
        [{"block_type": "rnn", "dim_hidden": 4, "dim_proj": 2, "rnn_type": "foo"}],
    ],
)
def test_wrong_block_arguments(body_conf):
    with pytest.raises(ValueError):
        _ = Encoder(8, body_conf)


@pytest.mark.parametrize(
    "input_conf, inputs",
    [
        ({"block_type": "conv2d", "subsampling_factor": 2}, [2, 2]),
        ({"block_type": "conv2d", "subsampling_factor": 4}, [6, 6]),
        ({"block_type": "conv2d", "subsampling_factor": 6}, [10, 5]),
    ],
)
def test_too_short_utterance(input_conf, inputs):
    input_size = 20

    body_conf = [{"block_type": "conformer", "dim_hidden": 4, "dim_linear": 2}]

    encoder = Encoder(input_size, body_conf, input_conf=input_conf)

    sequence = torch.randn(len(inputs), inputs[0], input_size, requires_grad=True)
    sequence_len = torch.tensor(inputs, dtype=torch.long)

    with pytest.raises(TooShortUttError):
        _ = encoder(sequence, sequence_len)


def test_wrong_subsampling_factor():
    input_conf = {"block_type": "conv2d", "subsampling_factor": 8}
    body_conf = [{"block_type": "conformer", "dim_hidden": 4, "dim_linear": 2}]

    with pytest.raises(ValueError):
        _ = Encoder(8, body_conf, input_conf=input_conf)


@pytest.mark.parametrize(
    "body_conf",
    [
        [
            {"block_type": "conv1d", "dim_output": 8, "kernel_size": 1},
            {"block_type": "conformer", "dim_hidden": 4, "dim_linear": 2},
        ],
        [
            {"block_type": "conformer", "dim_hidden": 8, "dim_linear": 2},
            {"block_type": "conformer", "dim_hidden": 4, "dim_linear": 2},
        ],
    ],
)
def test_wrong_block_io(body_conf):
    with pytest.raises(ValueError):
        _ = Encoder(8, body_conf)
