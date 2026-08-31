"""End-to-end forward coverage: cfm, dit, modules, rotary and utils in one call."""

import pytest
import torch

from espnet3.systems.tts.f5_tts.f5tts import F5TTS

FS = 24000
N_MELS = 100
MODEL_CONF = dict(
    hidden_size=32,
    depth=1,
    attention_heads=2,
    attention_head_size=16,
    feed_forward_multiplier=1,
    text_embedding_size=16,
    convolution_layers=1,
    ode_solver_method="euler",
)
FEATS_CONF = dict(
    fs=FS,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=N_MELS,
    mel_spec_type="vocos",
)
VOCAB = ["<blank>", "<unk>", "a", "b", "c", "<sos/eos>"]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return F5TTS(
        token_list=list(VOCAB),
        feats_extract_config=FEATS_CONF,
        **MODEL_CONF,
    )


def _batch(batch_size=2, n_tokens=6, n_samples=FS // 2):
    return dict(
        text=torch.randint(2, len(VOCAB), (batch_size, n_tokens)),
        text_lengths=torch.full((batch_size,), n_tokens, dtype=torch.long),
        speech=torch.randn(batch_size, n_samples),
        speech_lengths=torch.full((batch_size,), n_samples, dtype=torch.long),
    )


def test_forward_returns_a_finite_scalar_loss(model):
    loss, stats, weight = model(**_batch())
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "loss" in stats


def test_loss_backpropagates_to_the_backbone(model):
    loss, _, _ = model(**_batch())
    loss.backward()
    grads = [
        p.grad
        for p in model.cfm.transformer.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads, "no gradient reached the DiT backbone"
    assert any(g.abs().sum() > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_forward_handles_ragged_lengths(model):
    """Padded batches must mask correctly rather than NaN."""
    batch = _batch(batch_size=3, n_tokens=6)
    batch["text_lengths"] = torch.tensor([6, 4, 2])
    batch["speech_lengths"] = torch.tensor([FS // 2, FS // 3, FS // 4])
    loss, _, _ = model(**batch)
    assert torch.isfinite(loss)


def test_batch_of_one_works(model):
    loss, _, _ = model(**_batch(batch_size=1))
    assert torch.isfinite(loss)


def test_collect_feats_shape_matches_the_mel_front_end(model):
    out = model.collect_feats(**_batch(batch_size=2))
    assert out["feats"].shape[0] == 2
    assert out["feats"].shape[2] == N_MELS
    assert torch.isfinite(out["feats"]).all()


def test_loss_is_seed_reproducible(model):
    torch.manual_seed(1234)
    first, _, _ = model(**_batch())
    torch.manual_seed(1234)
    second, _, _ = model(**_batch())
    assert first.item() == pytest.approx(second.item())


def test_mel_dim_is_wired_from_the_feature_extractor(model):
    assert model.mel_dim == N_MELS
    assert model.feats_extract.output_size == N_MELS
