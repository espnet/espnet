"""Layer-level unit tests for the bundled PEFT backends."""

import pytest
import torch

from espnet2.layers.lora import (
    LINEAR_BACKENDS,
    DoraLinear,
    Embedding,
    Linear,
    LoRALayer,
    PiSSALinear,
    SSVDLinear,
    SVFTLinear,
)

# Default tiny dimensions for fast tests.
IN_FEATURES = 16
OUT_FEATURES = 16  # square so SSVD's apply_rotation path doesn't matter
RANK = 4
ALPHA = 4
BATCH = 2


def _adapter_param_names():
    """Substrings that identify trainable adapter parameters."""
    return ("lora_A", "lora_B", "lora_m", "s", "gate", "K_vec", "m_entries")


def _make_layer(cls, **extra):
    kwargs = dict(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        r=RANK,
        lora_alpha=ALPHA,
    )
    kwargs.update(extra)
    return cls(**kwargs)


@pytest.mark.parametrize("name,cls", sorted(LINEAR_BACKENDS.items()))
def test_backend_forward_shape(name, cls):
    layer = _make_layer(cls)
    x = torch.randn(BATCH, IN_FEATURES)
    y = layer(x)
    assert y.shape == (BATCH, OUT_FEATURES), f"{name}: unexpected shape {y.shape}"


@pytest.mark.parametrize("name,cls", sorted(LINEAR_BACKENDS.items()))
def test_backend_is_lora_layer(name, cls):
    layer = _make_layer(cls)
    assert isinstance(layer, LoRALayer), f"{name} must inherit LoRALayer"


@pytest.mark.parametrize("name,cls", sorted(LINEAR_BACKENDS.items()))
def test_backend_pretrained_weight_frozen(name, cls):
    layer = _make_layer(cls)
    assert (
        layer.weight.requires_grad is False
    ), f"{name}: pretrained .weight must be frozen by the adapter"


@pytest.mark.parametrize("name,cls", sorted(LINEAR_BACKENDS.items()))
def test_backend_has_trainable_adapter_params(name, cls):
    layer = _make_layer(cls)
    trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
    assert trainable, f"{name}: no trainable parameters found"
    # At least one trainable parameter should be a recognised adapter param.
    assert any(
        any(tok in n for tok in _adapter_param_names()) for n in trainable
    ), f"{name}: no adapter-like trainable parameter found in {trainable}"


@pytest.mark.parametrize("name,cls", sorted(LINEAR_BACKENDS.items()))
def test_backend_gradient_flows_to_adapter(name, cls):
    layer = _make_layer(cls)
    x = torch.randn(BATCH, IN_FEATURES, requires_grad=False)
    y = layer(x).sum()
    y.backward()
    # The pretrained weight must NOT receive a grad.
    assert (
        layer.weight.grad is None
    ), f"{name}: frozen .weight should not receive a gradient"
    # At least one adapter parameter must have a non-zero gradient.
    has_grad = False
    for n, p in layer.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is not None and torch.any(p.grad != 0):
            has_grad = True
            break
    assert has_grad, f"{name}: no adapter parameter received a non-zero gradient"


def test_lora_linear_zero_init_is_identity():
    """Vanilla LoRA: at init, BA = 0 so the layer equals the base nn.Linear."""
    layer = _make_layer(Linear)
    x = torch.randn(BATCH, IN_FEATURES)
    y_lora = layer(x)
    y_ref = torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.allclose(
        y_lora, y_ref, atol=1e-6
    ), "LoRA forward at init must equal the base linear because B is zero-initialised."


def test_pissa_init_matches_pretrained():
    """PiSSA layer must equal the base linear at init.

    At init ``lora_A/lora_B`` equal the frozen ``A0/B0`` factors, so the
    delta ``(BA - B0A0) * scaling`` is exactly zero.
    """
    layer = _make_layer(PiSSALinear)
    x = torch.randn(BATCH, IN_FEATURES)
    y_pissa = layer(x)
    y_ref = torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.allclose(
        y_pissa, y_ref, atol=1e-5
    ), "PiSSA must be an identity adapter at init (delta = 0)."


def test_dora_magnitude_initialised_to_pretrained_norm():
    layer = _make_layer(DoraLinear)
    # apply_m fires on first forward.
    layer(torch.randn(BATCH, IN_FEATURES))
    expected = torch.linalg.norm(layer.weight, dim=1).unsqueeze(1)
    assert torch.allclose(layer.lora_m.data, expected, atol=1e-6), (
        "DoRA magnitude must be initialised to the row-wise L2 norm of the "
        "pretrained weight."
    )


def test_svft_band_indices_have_expected_count():
    """SVFT-banded with off_diag=d has (2d+1)·n − d·(d+1) band entries."""
    off_diag = 1
    layer = _make_layer(SVFTLinear, off_diag=off_diag)
    n = min(IN_FEATURES, OUT_FEATURES)
    expected = n * (2 * off_diag + 1) - off_diag * (off_diag + 1)
    assert (
        layer.num_banded_params == expected
    ), f"SVFT band count mismatch: got {layer.num_banded_params}, expected {expected}"
    assert layer.m_entries.shape == (expected,)


def test_svft_invalid_off_diag_raises():
    with pytest.raises(ValueError):
        _make_layer(SVFTLinear, off_diag=-1)


def test_ssvd_apply_svd_populates_buffers():
    layer = _make_layer(SSVDLinear)
    assert bool(layer.svd_initialized) is False
    layer(torch.randn(BATCH, IN_FEATURES))
    assert bool(layer.svd_initialized) is True
    # Reconstruct W from U S V and compare to the original weight.
    reconstructed = layer.u @ torch.diag(layer.s_pre) @ layer.v
    assert torch.allclose(
        reconstructed, layer.weight, atol=1e-4
    ), "SSVD's cached u/s_pre/v must reconstruct the pretrained weight."


def test_ssvd_rotation_ratio_sets_k_trainable():
    layer = SSVDLinear(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        rotation_ratio=0.5,
        lora_alpha=ALPHA,
    )
    assert layer.k_trainable == int(min(IN_FEATURES, OUT_FEATURES) * 0.5)


@pytest.mark.parametrize("rotation_ratio", [0.0, -0.1, 1.5])
def test_ssvd_invalid_rotation_ratio_raises(rotation_ratio):
    with pytest.raises(ValueError):
        SSVDLinear(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            rotation_ratio=rotation_ratio,
            lora_alpha=ALPHA,
        )


def test_ssvd_without_rank_or_ratio_raises():
    with pytest.raises(ValueError):
        SSVDLinear(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            lora_alpha=ALPHA,
        )


def test_ssvd_supports_non_square_layers_in_both_orientations():
    # tall layer (out > in)
    tall = SSVDLinear(in_features=8, out_features=16, r=4, lora_alpha=ALPHA)
    tall(torch.randn(BATCH, 8))
    # wide layer (out < in)
    wide = SSVDLinear(in_features=16, out_features=8, r=4, lora_alpha=ALPHA)
    wide(torch.randn(BATCH, 16))


def test_embedding_forward_shape_and_freeze():
    emb = Embedding(num_embeddings=10, embedding_dim=8, r=RANK, lora_alpha=ALPHA)
    assert emb.weight.requires_grad is False
    assert emb.lora_A.requires_grad and emb.lora_B.requires_grad
    idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    out = emb(idx)
    assert out.shape == (2, 3, 8)


@pytest.mark.parametrize("name,cls", sorted(LINEAR_BACKENDS.items()))
def test_backend_eval_before_forward_keeps_weight(name, cls):
    """eval() before any forward must not disturb the (pretrained) weight.

    create_lora_adapter calls model.eval() right after module replacement,
    before checkpoints are loaded; backends with lazily initialized factors
    must not merge from uninitialized state.
    """
    layer = _make_layer(cls)
    w0 = layer.weight.detach().clone()
    layer.eval()
    assert torch.allclose(layer.weight, w0), f"{name}: eval() changed the weight"
    layer.train()
    assert torch.allclose(layer.weight, w0), f"{name}: train() changed the weight"


@pytest.mark.parametrize("name,cls", sorted(LINEAR_BACKENDS.items()))
def test_backend_train_eval_merge_round_trip(name, cls):
    """Merged (eval) and unmerged (train) paths must produce the same output."""
    torch.manual_seed(0)
    layer = _make_layer(cls)
    x = torch.randn(BATCH, IN_FEATURES)
    y_train = layer(x)

    layer.eval()
    y_eval = layer(x)
    assert torch.allclose(
        y_train, y_eval, atol=1e-4
    ), f"{name}: merge changed the output"

    layer.train()
    y_train2 = layer(x)
    assert torch.allclose(
        y_train2, y_eval, atol=1e-4
    ), f"{name}: unmerge changed the output"
