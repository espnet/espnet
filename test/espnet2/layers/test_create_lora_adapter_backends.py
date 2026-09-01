"""Tests for ``create_lora_adapter``'s ``adapter_type`` backend dispatch.

These tests live in a separate file from ``test_create_adapter_fn.py`` so
they do not get skipped when ``transformers`` / ``s3prl`` are unavailable
(those packages are needed by the Houlsby tests, not by LoRA backends).
"""

import pytest

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.layers.create_adapter_fn import create_lora_adapter
from espnet2.layers.lora import (
    DoraLinear,
)
from espnet2.layers.lora import Linear as LoraLinear
from espnet2.layers.lora import PiSSALinear, SSVDLinear, SVFTLinear


def init_decoder_model():
    return TransformerDecoder(
        vocab_size=10,
        encoder_output_size=40,
        attention_heads=4,
        linear_units=40,
        num_blocks=2,
        input_layer="embed",
    )


@pytest.mark.parametrize(
    "adapter_type, expected_cls, extra_kwargs",
    [
        ("lora", LoraLinear, {}),
        ("dora", DoraLinear, {}),
        ("pissa", PiSSALinear, {}),
        ("svft", SVFTLinear, {}),
        ("ssvd", SSVDLinear, {"rotation_ratio": 0.5}),
    ],
)
def test_create_lora_adapter_backend_dispatch(adapter_type, expected_cls, extra_kwargs):
    model = init_decoder_model()
    create_lora_adapter(
        model=model,
        rank=2,
        alpha=4,
        target_modules=["linear_q"],
        adapter_type=adapter_type,
        **extra_kwargs,
    )
    linear_q = model.decoders[0].self_attn.linear_q
    assert isinstance(
        linear_q, expected_cls
    ), f"adapter_type={adapter_type!r} did not produce {expected_cls.__name__}"
    # At least one adapter parameter must be trainable. (Freezing the
    # pretrained .weight is the caller's responsibility -- e.g. via the
    # YAML `freeze_param` field or `mark_only_lora_as_trainable`.)
    trainable = [n for n, p in linear_q.named_parameters() if p.requires_grad]
    assert any(
        n != "weight" and n != "bias" for n in trainable
    ), f"{adapter_type}: no adapter parameter is trainable: {trainable}"


def test_create_lora_adapter_unknown_backend_raises():
    model = init_decoder_model()
    with pytest.raises(ValueError, match="Unsupported adapter_type"):
        create_lora_adapter(
            model=model,
            rank=2,
            alpha=4,
            target_modules=["linear_q"],
            adapter_type="not_a_real_backend",
        )


def test_create_lora_adapter_ssvd_forwards_rotation_ratio():
    model = init_decoder_model()
    create_lora_adapter(
        model=model,
        rank=0,
        alpha=4,
        target_modules=["linear_q"],
        adapter_type="ssvd",
        rotation_ratio=0.25,
    )
    linear_q = model.decoders[0].self_attn.linear_q
    assert isinstance(linear_q, SSVDLinear)
    # rotation_ratio=0.25 of min(40, 40)=40 -> k_trainable=10
    assert linear_q.k_trainable == 10
