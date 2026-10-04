import numpy as np
import pytest
import torch

from espnet2.universa.base import UniversaBase


@pytest.mark.parametrize("multi_branch", [False, True])
@pytest.mark.parametrize("pooling_type", ["mean", "channel_attention"])
@pytest.mark.parametrize("use_normalize", [False, True])
def test_training_and_inference(make_model, multi_branch, pooling_type, use_normalize):
    model = make_model(
        multi_branch=multi_branch,
        pooling_type=pooling_type,
        use_normalize=use_normalize,
    )
    audio = torch.randn(3, 10, 8)
    lengths = torch.tensor([10, 8, 6])
    # Entirely absent metrics and individually missing labels are both supported.
    metrics = {"mos": torch.tensor([1.0, -100.0, 3.0])}
    loss, stats, weight = model(audio, lengths, metrics)
    assert torch.isfinite(loss) and loss > 0
    torch.testing.assert_close(loss, stats["loss"])
    assert weight == 3
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    model.eval()
    result = model.inference(audio, lengths)
    assert set(result) == {"mos", "wer", "encoded_feat"}
    assert all(np.isfinite(result[key]).all() for key in metrics)
    # Padding must not affect a shorter utterance's predictions.
    audio[2, 6:] = 1000
    padded = model.inference(audio, lengths)
    np.testing.assert_allclose(result["mos"], padded["mos"], atol=1e-5)


@pytest.mark.parametrize("projector_type", ["linear", "xvector"])
@pytest.mark.parametrize("multi_branch", [False, True])
def test_optional_reference_and_projector(make_model, projector_type, multi_branch):
    model = make_model(
        use_ref_audio=True, projector_type=projector_type, multi_branch=multi_branch
    )
    result = model.inference(torch.randn(2, 10, 8), torch.tensor([10, 8]))
    assert result["mos"].shape == (2,)
    assert result["encoded_feat"].shape[-1] == 16


def test_legacy_embedding_dim_remains_ignored():
    model = UniversaBase(
        input_size=8,
        metric2id={"mos": 0},
        embedding_dim=256,
        use_ref_audio=False,
        use_ref_text=False,
        audio_encoder_params=dict(
            num_blocks=1, attention_heads=2, linear_units=16, input_layer="linear"
        ),
    )
    assert model.embedding_size == 512


@pytest.mark.parametrize("multi_branch", [False, True])
@pytest.mark.parametrize("weights", [{"mos": 0.0, "wer": 1.0}, {0: 0.0, 1: 1.0}])
def test_named_loss_weights(make_model, multi_branch, weights):
    model = make_model(multi_branch=multi_branch, loss_weights=weights)
    # Only the zero-weight target is present, so it must not train the predictor.
    loss, _, _ = model(
        torch.randn(2, 10, 8), torch.tensor([10, 8]), {"mos": torch.tensor([2.0, 3.0])}
    )
    assert loss.item() == 0
    loss.backward()
    assert all(
        torch.count_nonzero(p.grad) == 0
        for p in model.parameters()
        if p.grad is not None
    )


@pytest.mark.parametrize(
    "metric2id",
    [{}, {"mos": 1}, {"mos": 0, "wer": 0}, {"mos": "0"}, {"mos": 0.0}, {"mos": False}],
)
def test_invalid_metric_ids(make_model, metric2id):
    with pytest.raises(ValueError, match="metric2id"):
        make_model(metric2id=metric2id)


@pytest.mark.parametrize("vocab_size", [None, 0, -1])
def test_reference_text_requires_vocabulary(make_model, vocab_size):
    with pytest.raises(ValueError, match="vocab_size"):
        make_model(use_ref_text=True, vocab_size=vocab_size)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_normalization_preserves_caller_audio(make_model, requires_grad):
    model = make_model(use_normalize=True, use_ref_audio=True).eval()
    audio = torch.randn(2, 10, 8, requires_grad=requires_grad)
    reference = torch.randn(2, 12, 8, requires_grad=requires_grad)
    original_audio, original_reference = (
        audio.detach().clone(),
        reference.detach().clone(),
    )
    with torch.set_grad_enabled(requires_grad):
        encoded, _ = model.encode(
            audio, torch.tensor([10, 8]), reference, torch.tensor([12, 9])
        )
        if requires_grad:
            encoded.square().sum().backward()
            assert audio.grad is not None and torch.isfinite(audio.grad).all()
            assert reference.grad is not None and torch.isfinite(reference.grad).all()
    torch.testing.assert_close(audio, original_audio)
    torch.testing.assert_close(reference, original_reference)
