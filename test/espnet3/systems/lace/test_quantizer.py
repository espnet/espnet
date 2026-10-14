"""Unit tests for espnet3.systems.lace.src.quantizer."""

import copy
from test.espnet3.systems.lace._codec_dummies import (
    HIDDEN_DIM,
    NUM_CODEBOOKS,
    build_tiny_codec_model,
)

import pytest
import torch

from espnet3.systems.lace.src.cosine_similarity import CosineSimilarityCompression
from espnet3.systems.lace.src.identity import IdentityCompression
from espnet3.systems.lace.src.quantizer import (
    CompressionResidualVectorQuantizer,
    RVQCompressionWrapper,
)
from espnet3.utils.task_utils import get_espnet_model

FRAME_RATE = 30


@pytest.fixture
def original_quantizer():
    return build_tiny_codec_model().codec.generator.quantizer


def build_wrapped(original_quantizer, compression_model=None, **kwargs):
    return CompressionResidualVectorQuantizer(
        copy.deepcopy(original_quantizer),
        compression_model or CosineSimilarityCompression(),
        **kwargs,
    )


def test_rejects_unknown_random_rate(original_quantizer):
    with pytest.raises(ValueError, match="random_rate"):
        build_wrapped(original_quantizer, random_rate="per_batch")


def test_rejects_quantizer_without_inner_vq():
    with pytest.raises(TypeError, match="ResidualVectorQuantizer"):
        CompressionResidualVectorQuantizer(torch.nn.Linear(2, 2), IdentityCompression())


def test_wraps_inner_vq_and_exposes_properties(original_quantizer):
    compression_model = IdentityCompression()
    wrapped = build_wrapped(original_quantizer, compression_model)
    assert isinstance(wrapped.rvq.vq, RVQCompressionWrapper)
    assert wrapped.compression_model is compression_model
    assert wrapped.num_codebooks == NUM_CODEBOOKS
    assert wrapped.rvq.vq.quantizer_dropout == 0.0
    assert wrapped.get_bandwidth_per_quantizer(
        FRAME_RATE
    ) == original_quantizer.get_bandwidth_per_quantizer(FRAME_RATE)
    assert wrapped.get_num_quantizers_for_bandwidth(FRAME_RATE, None) == NUM_CODEBOOKS


@pytest.mark.parametrize("training", [True, False])
def test_identity_compression_matches_unwrapped_quantizer(original_quantizer, training):
    # random_rate="none": sampling a rate would consume random numbers that
    # the unwrapped quantizer does not, and shift its dead-code resampling.
    wrapped = build_wrapped(
        original_quantizer, IdentityCompression(), random_rate="none", eval_rate=0.5
    )
    reference = copy.deepcopy(original_quantizer)
    wrapped.train(training)
    reference.train(training)
    x = torch.randn(2, HIDDEN_DIM, 12)
    torch.manual_seed(0)
    expected = reference(x, FRAME_RATE)
    torch.manual_seed(0)
    actual = wrapped(x, FRAME_RATE)
    assert len(actual) == len(expected) == 4
    for actual_item, expected_item in zip(actual, expected):
        assert torch.allclose(actual_item, expected_item)
    wrapped.eval()
    reference.eval()
    codes = wrapped.encode(x, FRAME_RATE)
    assert torch.equal(codes, reference.encode(x, FRAME_RATE))
    assert torch.allclose(wrapped.decode(codes), reference.decode(codes))


def test_state_dict_layout_round_trips(original_quantizer):
    wrapped = build_wrapped(original_quantizer)
    assert all(key.startswith("rvq.vq.rvq.") for key in wrapped.state_dict())
    assert len(wrapped.state_dict()) == len(original_quantizer.state_dict())
    for key, value in original_quantizer.state_dict().items():
        assert torch.equal(wrapped.state_dict()["rvq.vq.rvq." + key[3:]], value)
    # Wrapped-layout keys (a fine-tuned checkpoint) load strictly.
    wrapped.load_state_dict(copy.deepcopy(wrapped.state_dict()), strict=True)


def test_resolve_rate_precedence(original_quantizer):
    wrapped = build_wrapped(original_quantizer, eval_rate=0.5)
    wrapped.eval()
    assert wrapped._resolve_rate(2, "cpu", 0.9) == 0.9
    assert wrapped._resolve_rate(2, "cpu", None) == 0.5
    wrapped.set_inference_rate(0.3)
    assert wrapped._resolve_rate(2, "cpu", None) == 0.3
    wrapped.reset_inference_rate()
    assert wrapped._resolve_rate(2, "cpu", None) == 0.5


@pytest.mark.parametrize(
    "random_rate, shape",
    [("per_quantizer", (NUM_CODEBOOKS, 3)), ("per_sample", (3,)), ("none", None)],
)
def test_training_rate_sampling(original_quantizer, random_rate, shape):
    wrapped = build_wrapped(
        original_quantizer, min_rate=0.2, max_rate=0.6, random_rate=random_rate
    )
    wrapped.train()
    rate = wrapped._resolve_rate(3, "cpu", None)
    if shape is None:
        assert rate is None
    else:
        assert rate.shape == shape
        assert rate.min() >= 0.2 and rate.max() <= 0.6


@pytest.mark.parametrize("random_rate", ["per_quantizer", "per_sample", "none"])
def test_training_forward_backward(original_quantizer, random_rate):
    wrapped = build_wrapped(original_quantizer, random_rate=random_rate)
    wrapped.train()
    x = torch.randn(2, HIDDEN_DIM, 12, requires_grad=True)
    quantized, codes, bandwidth, commit_loss = wrapped(x, FRAME_RATE)
    assert quantized.shape == x.shape
    assert codes.shape == (NUM_CODEBOOKS, 2, 12)
    (quantized.sum() + commit_loss).backward()
    assert x.grad is not None


def test_lower_rate_gives_fewer_segments(original_quantizer):
    wrapped = build_wrapped(original_quantizer)
    wrapped.eval()
    torch.manual_seed(0)
    x = torch.randn(2, HIDDEN_DIM, 20)
    counts = []
    for rate in [0.05, 0.95]:
        codes, residuals, segment_indices = wrapped.encode_with_segments(
            x, FRAME_RATE, rate=rate
        )
        assert codes.shape == (NUM_CODEBOOKS, 2, 20)
        assert residuals.shape == (NUM_CODEBOOKS, 2, HIDDEN_DIM, 20)
        assert len(segment_indices) == NUM_CODEBOOKS
        counts.append(sum(int(idx.max(dim=1).values.sum()) for idx in segment_indices))
    assert counts[0] < counts[1]


def test_encode_layer_range_and_bandwidth(original_quantizer):
    wrapped = build_wrapped(original_quantizer, IdentityCompression())
    wrapped.eval()
    x = torch.randn(1, HIDDEN_DIM, 10)
    assert wrapped.encode(x, FRAME_RATE, st=2).shape == (NUM_CODEBOOKS - 2, 1, 10)
    one_layer_bandwidth = wrapped.get_bandwidth_per_quantizer(FRAME_RATE)
    assert wrapped.encode(x, FRAME_RATE, one_layer_bandwidth).shape == (1, 1, 10)


def test_anchor_start_layer_restricts_later_layers(original_quantizer):
    wrapped = build_wrapped(
        original_quantizer, CosineSimilarityCompression(max_tokens_per_group=None)
    )
    wrapped.eval()
    torch.manual_seed(0)
    x = torch.randn(2, HIDDEN_DIM, 24)
    wrapped.set_inference_rate(0.5)
    wrapped.set_inference_anchor_start_layer(1)
    _, _, anchored = wrapped.encode_with_segments(x, FRAME_RATE)
    wrapped.reset_inference_anchor_start_layer()
    _, _, free = wrapped.encode_with_segments(x, FRAME_RATE)
    wrapped.reset_inference_rate()

    def boundaries(segment_idx):
        boundary = torch.zeros_like(segment_idx, dtype=torch.bool)
        boundary[:, 1:] = segment_idx[:, 1:] != segment_idx[:, :-1]
        return boundary

    anchor = boundaries(anchored[0])
    assert torch.equal(anchored[0], free[0])
    for segment_idx in anchored[1:]:
        assert (boundaries(segment_idx) & ~anchor).sum() == 0
    assert any((boundaries(idx) & ~anchor).sum() > 0 for idx in free[1:])
    # An explicit argument is equivalent to the inference-time setter.
    _, _, explicit = wrapped.encode_with_segments(
        x, FRAME_RATE, rate=0.5, anchor_start_layer=1
    )
    for explicit_idx, anchored_idx in zip(explicit, anchored):
        assert torch.equal(explicit_idx, anchored_idx)


def test_quantizer_dropout_codec_returns_five_values():
    torch.manual_seed(0)
    model = get_espnet_model(
        "espnet2.tasks.gan_codec.GANCodecTask", {"codec": "dac", "codec_conf": {}}
    )
    generator = model.codec.generator
    wrapped = CompressionResidualVectorQuantizer(
        generator.quantizer, CosineSimilarityCompression(), eval_rate=0.5
    )
    assert wrapped.rvq.vq.quantizer_dropout
    with torch.no_grad():
        x = generator.encoder(torch.randn(2, 1, 2400))
    for training in [True, False]:
        wrapped.train(training)
        output = wrapped(x, 75, anchor_start_layer=1)
        assert len(output) == 5
        assert output[0].shape == x.shape
