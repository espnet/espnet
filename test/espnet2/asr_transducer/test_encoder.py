import pytest
import torch

from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.encoder.modules.attention import (
    RelPositionMultiHeadedAttention,
)
from espnet2.asr_transducer.utils import (
    TooShortUttError,
    make_chunk_mask,
    make_source_mask,
)


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
                "num_left_chunks": 2,
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
                "num_left_chunks": 0,
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

    # Note (b-flo): For each test with Conformer blocks, we do the same testing with
    # Branchformer and E-Branchformer blocks instead.
    # The tests will be redesigned, for now we avoid writing too many configs.
    _swap = False

    branchformer_conf = body_conf[:]
    ebranchformer_conf = body_conf[:]

    for i, b in enumerate(body_conf):
        if b["block_type"] == "conformer":
            branchformer_conf[i]["block_type"] = "branchformer"
            ebranchformer_conf[i]["block_type"] = "ebranchformer"

            if _swap is False:
                _swap = True

    if _swap:
        branchformer_encoder = Encoder(
            input_size, branchformer_conf, input_conf=input_conf, main_conf=main_conf
        )
        _ = branchformer_encoder(sequence, sequence_len)

        ebranchformer_encoder = Encoder(
            input_size, ebranchformer_conf, input_conf=input_conf, main_conf=main_conf
        )
        _ = ebranchformer_encoder(sequence, sequence_len)


@pytest.mark.parametrize(
    "conv_conf",
    [
        {"kernel_size": 3},
        {"kernel_size": 2, "dilation": 2},
        {"kernel_size": 3, "stride": 2},
    ],
)
def test_conv1d_block_output_lengths(conv_conf):
    body_conf = [
        {"block_type": "conv1d", "output_size": 4, **conv_conf},
        {
            "block_type": "conformer",
            "hidden_size": 4,
            "linear_size": 2,
            "conv_mod_kernel_size": 3,
        },
    ]
    encoder = Encoder(20, body_conf, {"conv_size": 4}).eval()

    x = torch.randn(2, 60, 20)

    with torch.no_grad():
        out, out_lens = encoder(x, torch.tensor([60, 60]))

    # The lengths cannot be larger than the number of frames that were produced.
    assert out_lens.tolist() == [out.size(1)] * 2


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
        [{"block_type": "branchformer", "hidden_size": 4, "linear_size": 2}],
        [{"block_type": "conformer", "hidden_size": 4}],
        [{"block_type": "conformer", "hidden_size": 4, "linear_size": 2}],
        [{"block_type": "conv1d"}],
        [{"block_type": "conv1d", "output_size": 8}, {}],
        [{"block_type": "ebranchformer", "hidden_size": 4}],
        [{"block_type": "ebranchformer", "hidden_size": 4, "linear_size": 2}],
    ],
)
def test_wrong_block_arguments(body_conf):
    with pytest.raises(ValueError):
        _ = Encoder(8, body_conf)


@pytest.mark.parametrize(
    "input_conf, inputs",
    [
        ({"subsampling_factor": 2}, [2, 2]),
        ({"subsampling_factor": 4}, [6, 6]),
        ({"subsampling_factor": 6}, [10, 5]),
        ({"vgg_like": True}, [6, 6]),
        ({"vgg_like": True, "subsampling_factor": 6}, [10, 5]),
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
    "input_conf, body_conf",
    [
        (
            {"subsampling_factor": 8},
            [
                {
                    "block_type": "branchformer",
                    "hidden_size": 4,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                }
            ],
        ),
        (
            {"vgg_like": True, "subsampling_factor": 8},
            [
                {
                    "block_type": "conformer",
                    "hidden_size": 4,
                    "linear_size": 2,
                    "conv_mod_kernel_size": 1,
                }
            ],
        ),
    ],
)
def test_wrong_subsampling_factor(input_conf, body_conf):
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
            {"block_type": "conv1d", "output_size": 8, "kernel_size": 1},
            {
                "block_type": "ebranchformer",
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
                "block_type": "ebranchformer",
                "hidden_size": 8,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
            {
                "block_type": "ebranchformer",
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
        [
            {
                "block_type": "conformer",
                "hidden_size": 8,
                "conv_mod_kernel_size": 2,
                "linear_size": 2,
            },
            {
                "block_type": "ebranchformer",
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


@pytest.mark.parametrize(
    "input_conf",
    [
        {"subsampling_factor": 2, "conv_size": 4},
        {"subsampling_factor": 4, "conv_size": 4},
        {"subsampling_factor": 6, "conv_size": 4},
        {"vgg_like": True, "subsampling_factor": 4, "conv_size": 4},
        {"vgg_like": True, "subsampling_factor": 6, "conv_size": 4},
    ],
)
def test_input_block_output_lengths(input_conf):
    body_conf = [
        {
            "block_type": "conformer",
            "hidden_size": 4,
            "linear_size": 2,
            "conv_mod_kernel_size": 3,
        }
    ]
    encoder = Encoder(20, body_conf, input_conf).eval()

    feat_lens = [60, 47, 33]

    x = torch.randn(len(feat_lens), max(feat_lens), 20)
    for i, feat_len in enumerate(feat_lens):
        x[i, feat_len:] = 0.0

    with torch.no_grad():
        batch_out, batch_lens = encoder(x, torch.tensor(feat_lens))

        for i, feat_len in enumerate(feat_lens):
            # The output length of an utterance does not depend on its batch.
            out, out_len = encoder(x[i : i + 1, :feat_len], torch.tensor([feat_len]))

            assert out.size(1) == out_len.item()
            assert batch_lens[i].item() == out_len.item()

    assert batch_out.size(1) == batch_lens.max().item()


def _build_sdpa_encoder(block_type, simplified_att_score, use_sdpa, streaming=False):
    """Build identical small encoders for the default and SDPA paths."""
    torch.manual_seed(0)
    return Encoder(
        20,
        [
            {
                "block_type": block_type,
                "hidden_size": 16,
                "linear_size": 32,
                "heads": 2,
                "conv_mod_kernel_size": 3,
                "num_blocks": 2,
                "att_dropout_rate": 0.2,
            }
        ],
        input_conf={"conv_size": 4},
        main_conf={
            "simplified_att_score": simplified_att_score,
            "use_sdpa": use_sdpa,
            "dynamic_chunk_training": streaming,
            "pos_enc_max_len": 128,
        },
    ).eval()


@pytest.mark.parametrize("block_type", ["conformer", "branchformer", "ebranchformer"])
@pytest.mark.parametrize("simplified_att_score", [False, True])
def test_use_sdpa_reaches_the_attention_and_matches_the_default(
    block_type, simplified_att_score
):
    """Every block receives use_sdpa and preserves valid outputs with padding."""
    default = _build_sdpa_encoder(block_type, simplified_att_score, False)
    sdpa = _build_sdpa_encoder(block_type, simplified_att_score, True)
    for encoder, use_sdpa in [(default, False), (sdpa, True)]:
        assert len(encoder.encoders.blocks) == 2
        for block in encoder.encoders.blocks:
            assert isinstance(block.self_att, RelPositionMultiHeadedAttention)
            assert block.self_att.use_sdpa is use_sdpa

    x = torch.randn(2, 80, 20)
    lengths = torch.tensor([80, 55])
    x[1, 55:] = 0.0
    with torch.no_grad():
        expected, expected_lengths = default(x, lengths)
        actual, actual_lengths = sdpa(x, lengths)

    assert actual.shape == expected.shape
    assert torch.equal(actual_lengths, expected_lengths)
    assert torch.allclose(actual, expected, atol=1e-5)
    assert all(block.self_att.attn is None for block in sdpa.encoders.blocks)


@pytest.mark.parametrize("block_type", ["conformer", "branchformer", "ebranchformer"])
@pytest.mark.parametrize("simplified_att_score", [False, True])
def test_use_sdpa_streaming_matches_the_default(block_type, simplified_att_score):
    """Chunk forwarding matches with both empty and populated left-context caches."""
    default = _build_sdpa_encoder(block_type, simplified_att_score, False, True)
    sdpa = _build_sdpa_encoder(block_type, simplified_att_score, True, True)
    x = torch.randn(1, 78, 20)
    left_context = 5
    default.reset_cache(left_context, x.device)
    sdpa.reset_cache(left_context, x.device)
    processed_frames = torch.tensor([0])

    with torch.no_grad():
        for start in [0, 24, 48]:
            chunk = x[:, start : start + 30]
            lengths = torch.tensor([chunk.size(1)])
            expected = default.chunk_forward(
                chunk, lengths, processed_frames, left_context=left_context
            )
            actual = sdpa.chunk_forward(
                chunk, lengths, processed_frames, left_context=left_context
            )
            assert actual.shape == expected.shape
            assert torch.allclose(actual, expected, atol=1e-5)
            for block in default.encoders.blocks:
                assert block.self_att.attn.size(-1) == expected.size(1) + left_context
            processed_frames += expected.size(1)


@pytest.mark.parametrize("simplified_att_score", [False, True])
def test_use_sdpa_with_chunk_and_padding_masks(simplified_att_score):
    """SDPA combines dynamic chunk and padding masks for both score variants."""
    torch.manual_seed(0)
    default = RelPositionMultiHeadedAttention(
        2, 16, dropout_rate=0.2, simplified_attention_score=simplified_att_score
    ).eval()
    assert default.use_sdpa is False
    torch.manual_seed(0)
    sdpa = RelPositionMultiHeadedAttention(
        2,
        16,
        dropout_rate=0.2,
        simplified_attention_score=simplified_att_score,
        use_sdpa=True,
    ).eval()
    x = torch.randn(2, 12, 16)
    pos_enc = torch.randn(1, 23, 16)
    mask = make_source_mask(torch.tensor([12, 3]))
    chunk_mask = make_chunk_mask(12, 3, num_left_chunks=1)
    # Later chunks in the shorter utterance have fully masked padded queries.
    assert (chunk_mask.unsqueeze(0) | mask.unsqueeze(1)).all(-1).any()

    with torch.no_grad():
        expected = default(x, x, x, pos_enc, mask, chunk_mask=chunk_mask)
        actual = sdpa(x, x, x, pos_enc, mask, chunk_mask=chunk_mask)

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=1e-5)
    assert sdpa.attn is None


@pytest.mark.parametrize("simplified_att_score", [False, True])
def test_use_sdpa_training_backward(simplified_att_score):
    """Training with SDPA and nonzero dropout propagates finite gradients."""
    torch.manual_seed(0)
    attention = RelPositionMultiHeadedAttention(
        2,
        16,
        dropout_rate=0.2,
        simplified_attention_score=simplified_att_score,
        use_sdpa=True,
    ).train()
    x = torch.randn(2, 12, 16, requires_grad=True)
    pos_enc = torch.randn(1, 23, 16, requires_grad=True)
    mask = make_source_mask(torch.tensor([12, 9]))
    out = attention(x, x, x, pos_enc, mask)
    assert torch.isfinite(out).all()
    out.square().mean().backward()

    for tensor in [x, pos_enc, *attention.parameters()]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("simplified_att_score", [False, True])
def test_use_sdpa_training_gradients_match_the_default(simplified_att_score):
    """SDPA gradients match the default path with dropout disabled."""
    torch.manual_seed(0)
    default = RelPositionMultiHeadedAttention(
        2,
        16,
        dropout_rate=0.0,
        simplified_attention_score=simplified_att_score,
        use_sdpa=False,
    ).train()
    torch.manual_seed(0)
    sdpa = RelPositionMultiHeadedAttention(
        2,
        16,
        dropout_rate=0.0,
        simplified_attention_score=simplified_att_score,
        use_sdpa=True,
    ).train()
    x = torch.randn(2, 12, 16)
    pos_enc = torch.randn(1, 23, 16)
    x_default = x.clone().requires_grad_()
    x_sdpa = x.clone().requires_grad_()
    pos_enc_default = pos_enc.clone().requires_grad_()
    pos_enc_sdpa = pos_enc.clone().requires_grad_()
    mask = make_source_mask(torch.tensor([12, 3]))
    chunk_mask = make_chunk_mask(12, 3, num_left_chunks=1)

    expected = default(
        x_default, x_default, x_default, pos_enc_default, mask, chunk_mask=chunk_mask
    )
    actual = sdpa(x_sdpa, x_sdpa, x_sdpa, pos_enc_sdpa, mask, chunk_mask=chunk_mask)
    expected.square().mean().backward()
    actual.square().mean().backward()

    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.allclose(x_sdpa.grad, x_default.grad, atol=1e-5)
    assert torch.allclose(pos_enc_sdpa.grad, pos_enc_default.grad, atol=1e-5)
    for (name_default, p_default), (name_sdpa, p_sdpa) in zip(
        default.named_parameters(), sdpa.named_parameters()
    ):
        assert name_default == name_sdpa
        assert torch.allclose(p_sdpa.grad, p_default.grad, atol=1e-5)
