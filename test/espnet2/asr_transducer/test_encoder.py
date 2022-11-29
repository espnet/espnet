import pytest
import torch

from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.utils import TooShortUttError


@pytest.mark.parametrize(
    "input_conf, body_conf, main_conf",
    [
        (
            {"vgg_like": True, "susbsampling_factor": 4, "conv_size": 8},
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                    "num_blocks": 2,
                    "avg_eps": 1e-8,
                }
            ],
            {},
        ),
        (
            {"vgg_like": True, "susbsampling_factor": 6, "conv_size": 8},
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                    "num_blocks": 2,
                    "avg_eps": 1e-8,
                }
            ],
            {},
        ),
        (
            {"vgg_like": True, "subsampling_factor": 4},
            [
                {
                    "block_type": "conv1d",
                    "output_size": 4,
                    "kernel_size": 1,
                    "batch_norm": True,
                    "relu": True,
                }
            ],
            {},
        ),
        (
            {"vgg_like": True, "subsampling_factor": 6},
            [
                {
                    "block_type": "conv1d",
                    "output_size": 4,
                    "kernel_size": 1,
                    "batch_norm": True,
                    "relu": True,
                }
            ],
            {},
        ),
        (
            {},
            [
                {
                    "block_type": "conv1d",
                    "output_size": 8,
                    "kernel_size": 2,
                    "dilation": 2,
                },
                {
                    "block_type": "conformer",
                    "hidden_size": 8,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                    "num_blocks": 2,
                },
                {"block_type": "conv1d", "output_size": 4, "kernel_size": 1},
            ],
            {},
        ),
        (
            {"conv_size": (8, 4)},
            [{"block_type": "conv1d", "output_size": 4, "kernel_size": 1}],
            {},
        ),
        (
            {"conv_size": 4},
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                }
            ],
            {},
        ),
        (
            {"conv_size": 2},
            [
                {
                    "block_type": "conv1d",
                    "output_size": 4,
                    "kernel_size": 2,
                    "dilation": 2,
                    "batch_norm": True,
                    "relu": True,
                },
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                    "num_blocks": 2,
                },
            ],
            {},
        ),
        (
            {"conv_size": 2},
            [
                {
                    "block_type": "conv1d",
                    "output_size": 8,
                    "kernel_size": 2,
                },
                {
                    "block_type": "conformer",
                    "hidden_size": 8,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                    "num_blocks": 2,
                },
            ],
            {
                "dynamic_chunk_training": True,
                "norm_type": "scale_norm",
                "short_chunk_size": 30,
                "short_chunk_threshold": 0.01,
                "left_chunk_size": 2,
            },
        ),
        (
            {},
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 8,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                    "norm_eps": 1e-5,
                    "norm_partial": 0.8,
                    "conv_mod_norm_eps": 0.4,
                    "num_blocks": 2,
                },
            ],
            {
                "simplified_att_score": True,
                "norm_type": "rms_norm",
                "conv_mod_norm_type": "basic_norm",
                "dynamic_chunk_training": True,
                "short_chunk_size": 1,
                "left_chunk_size": 0,
            },
        ),
        (
            {},
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                }
            ],
            {"norm_type": "rms_norm"},
        ),
    ],
)
def test_encoder(input_conf, body_conf, main_conf):
    input_size = 8

    encoder = Encoder(input_size, body_conf, input_conf=input_conf, main_conf=main_conf)

    sequence = torch.randn(2, 30, input_size, requires_grad=True)
    sequence_len = torch.tensor([30, 18], dtype=torch.long)

    _ = encoder(sequence, sequence_len)

    # For each test with Conformer block, we do the same testing with Branchformer
    _branchformer_flag = False

    for b in body_conf:
        if b["block_type"] == "conformer":
            b["block_type"] = "branchformer"

            _branchformer_flag = True

    if _branchformer_flag:
        branchformer_encoder = Encoder(
            input_size, body_conf, input_conf=input_conf, main_conf=main_conf
        )

        _ = branchformer_encoder(sequence, sequence_len)


@pytest.mark.parametrize(
    "input_conf, body_conf",
    [
        ({}, [{}]),
        ({}, [{"block_type": "foo"}]),
    ],
)
def test_block_type(input_conf, body_conf):
    with pytest.raises(ValueError):
        _ = Encoder(8, body_conf, input_conf=input_conf)


@pytest.mark.parametrize(
    "body_conf",
    [
        [{"block_type": "branchformer", "hidden_size": 4}],
        [{"block_type": "conformer", "hidden_size": 4}],
        [{"block_type": "conv1d"}],
        [{"block_type": "conv1d", "output_size": 8}, {}],
        [{"block_type": "branchformer", "hidden_size": 4, "linear_size": 2}],
        [{"block_type": "conformer", "hidden_size": 4, "linear_size": 2}],
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
        ({"vgg_like": True}, [6, 6]),
    ],
)
def test_too_short_utterance(input_conf, inputs):
    input_size = 20

    body_conf = [
        {
            "block_type": "conformer",
            "hidden_size": 4,
            "linear_size": 2,
            "conv_mod_kernel_size": 3,
        }
    ]

    encoder = Encoder(input_size, body_conf, input_conf=input_conf)

    sequence = torch.randn(len(inputs), inputs[0], input_size, requires_grad=True)
    sequence_len = torch.tensor(inputs, dtype=torch.long)

    with pytest.raises(TooShortUttError):
        _ = encoder(sequence, sequence_len)


@pytest.mark.parametrize(
    "body_conf",
    [
        [
            {
                "block_type": "branchformer",
                "hidden_size": 4,
                "linear_size": 2,
                "conv_mod_kernel_size": 1,
            }
        ],
        [
            {
                "block_type": "conformer",
                "hidden_size": 4,
                "linear_size": 2,
                "conv_mod_kernel_size": 1,
            }
        ],
    ],
)
def test_wrong_subsampling_factor(body_conf):
    input_conf = {"block_type": "conv2d", "subsampling_factor": 8}

    with pytest.raises(ValueError):
        _ = Encoder(8, body_conf, input_conf=input_conf)


@pytest.mark.parametrize(
    "body_conf",
    [
        [
            {"block_type": "conv1d", "output_size": 8, "kernel_size": 1},
            {
                "block_type": "branchformer",
                "hidden_size": 4,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
        ],
        [
            {"block_type": "conv1d", "output_size": 8, "kernel_size": 1},
            {
                "block_type": "conformer",
                "hidden_size": 4,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
        ],
        [
            {
                "block_type": "branchformer",
                "hidden_size": 8,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
            {
                "block_type": "branchformer",
                "hidden_size": 4,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
        ],
        [
            {
                "block_type": "branchformer",
                "hidden_size": 8,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
            {
                "block_type": "conformer",
                "hidden_size": 4,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
        ],
        [
            {
                "block_type": "conformer",
                "hidden_size": 8,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
            {
                "block_type": "conformer",
                "hidden_size": 4,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
        ],
    ],
)
def test_wrong_block_io(body_conf):
    with pytest.raises(ValueError):
        _ = Encoder(8, body_conf)
