"""Tests for ``create_lora_adapter``'s ``adapter_type`` backend dispatch.

These tests live in a separate file from ``test_create_adapter_fn.py`` so
they do not get skipped when ``transformers`` / ``s3prl`` are unavailable
(those packages are needed by the Houlsby tests, not by LoRA backends).
"""

import pytest
import torch

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


BACKENDS = [
    ("lora", {}),
    ("dora", {}),
    ("pissa", {}),
    ("svft", {}),
    ("ssvd", {"rotation_ratio": 0.5}),
]


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = torch.nn.Linear(12, 12)

    def forward(self, x):
        return self.linear_q(x)


def _adapted(adapter_type, extra_kwargs):
    torch.manual_seed(999)  # random init, deliberately != the pretrained weights
    model = _Tiny()
    create_lora_adapter(
        model=model,
        rank=4,
        alpha=8,
        target_modules=["linear_q"],
        adapter_type=adapter_type,
        **extra_kwargs,
    )
    return model


@pytest.mark.parametrize("adapter_type,extra_kwargs", BACKENDS)
def test_backend_end_to_end_train_save_infer(adapter_type, extra_kwargs):
    """Walk the real ESPnet order: create_adapter -> init_param -> train -> infer.

    `AbsTask.build_model` calls `create_adapter` (which ends with
    `model.eval()`) *before* `init_param` is loaded, and `Trainer` saves the
    checkpoint while the model is still in eval mode. Every backend must come
    out of that sequence producing the same output it produced during training.
    """
    torch.manual_seed(0)
    pretrained = _Tiny().state_dict()
    x = torch.randn(3, 12)

    model = _adapted(adapter_type, extra_kwargs)
    dst = model.state_dict()
    dst.update(pretrained)  # what load_pretrained_model() does for init_param
    model.load_state_dict(dst)

    model.train()
    y_init = model(x)
    y_base = torch.nn.functional.linear(
        x, pretrained["linear_q.weight"], pretrained["linear_q.bias"]
    )
    assert torch.allclose(y_init, y_base, atol=1e-4), (
        f"{adapter_type}: the adapter must be an identity at init, i.e. it must "
        f"factorize the weight loaded by init_param, not the constructor's."
    )

    with torch.no_grad():  # stand in for a few optimizer steps
        for name, p in model.named_parameters():
            if not name.startswith("linear_q.weight") and not name.startswith(
                "linear_q.bias"
            ):
                p.add_(torch.randn_like(p) * 0.05)

    model.train()
    y_train = model(x).detach()
    model.eval()  # Trainer validates, then saves from eval mode
    y_eval = model(x).detach()
    assert torch.allclose(
        y_train, y_eval, atol=1e-4
    ), f"{adapter_type}: eval() changed the output"
    ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # A fresh inference process: build, adapt, load, eval.
    infer = _adapted(adapter_type, extra_kwargs)
    infer.load_state_dict(ckpt)
    infer.eval()
    assert torch.allclose(infer(x).detach(), y_eval, atol=1e-4), (
        f"{adapter_type}: inference output differs from training; the merge "
        f"bookkeeping did not survive the checkpoint round trip."
    )

    # Resuming training from the same checkpoint must also be consistent.
    resumed = _adapted(adapter_type, extra_kwargs)
    resumed.load_state_dict(ckpt)
    resumed.train()
    assert torch.allclose(
        resumed(x).detach(), y_train, atol=1e-4
    ), f"{adapter_type}: resumed training output differs"
