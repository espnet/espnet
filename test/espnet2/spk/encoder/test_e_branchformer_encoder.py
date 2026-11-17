import pytest
import torch

from espnet2.spk.encoder.e_branchformer_encoder import EBranchformerSpkEncoder
from espnet2.spk.pooling.chn_attn_stat_pooling import ChnAttnStatPooling
from espnet2.tasks.spk import encoder_choices


def _encoder(**kwargs):
    torch.manual_seed(0)
    conf = dict(
        output_size=8,
        attention_heads=2,
        cgmlp_linear_units=16,
        cgmlp_conv_kernel=3,
        linear_units=16,
        num_blocks=3,
        input_layer="conv2d2",
        merge_conv_kernel=3,
    )
    conf.update(kwargs)
    return EBranchformerSpkEncoder(16, **conf)


def test_aggregate_layer_concatenates_every_block():
    encoder = _encoder().eval()
    assert encoder.output_size() == 8 * 3
    with torch.no_grad():
        out = encoder(torch.randn(2, 20, 16))
    # conv2d2 keeps (20 - 1) // 2 - 2 = 7 frames; output is (batch, dim, time)
    assert out.shape == (2, 24, 7)


def test_without_aggregation_returns_the_last_block():
    encoder = _encoder(aggregate_layer=False).eval()
    assert encoder.output_size() == 8
    with torch.no_grad():
        out = encoder(torch.randn(2, 20, 16))
    assert out.shape == (2, 8, 7)


def test_dropped_layers_keep_the_output_size():
    encoder = _encoder(layer_drop_rate=1.0).train()
    out = encoder(torch.randn(2, 20, 16))
    assert out.shape == (2, 24, 7)


def test_feeds_channel_attentive_pooling():
    encoder = _encoder(use_ffn=True, macaron_ffn=True).eval()
    pooling = ChnAttnStatPooling(input_size=encoder.output_size()).eval()
    with torch.no_grad():
        emb = pooling(encoder(torch.randn(2, 20, 16)))
    assert emb.shape == (2, 2 * encoder.output_size())


def test_registered_for_the_spk_task():
    assert encoder_choices.get_class("e_branchformer") is EBranchformerSpkEncoder


def test_interctc_is_rejected():
    with pytest.raises(ValueError):
        _encoder(interctc_layer_idx=[1])


def test_too_short_input_raises():
    from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
        TooShortUttError,
    )

    with pytest.raises(TooShortUttError):
        _encoder().eval()(torch.randn(1, 3, 16))
