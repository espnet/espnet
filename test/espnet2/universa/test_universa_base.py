import numpy as np
import pytest
import torch

from espnet2.universa.base import UniversaBase
from espnet2.universa.base.loss import masked_l1_loss, masked_mse_loss


def make_model(**kwargs):
    options = dict(
        input_size=8,
        metric2id={"mos": 0, "wer": 1},
        use_ref_audio=False,
        use_ref_text=False,
        embedding_size=8,
        audio_encoder_params=dict(
            num_blocks=1,
            attention_heads=2,
            linear_units=16,
            input_layer="linear",
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
        ),
        cross_attention_params=dict(n_head=2, dropout_rate=0.0),
        use_mse=True,
    )
    options.update(kwargs)
    return UniversaBase(**options)


@pytest.mark.parametrize("multi_branch", [False, True])
@pytest.mark.parametrize("pooling_type", ["mean", "channel_attention"])
@pytest.mark.parametrize("use_normalize", [False, True])
def test_training_and_inference(multi_branch, pooling_type, use_normalize):
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


@pytest.mark.parametrize("loss_fn", [masked_l1_loss, masked_mse_loss])
def test_missing_labels_have_zero_loss_and_gradient(loss_fn):
    prediction = torch.randn(3, requires_grad=True)
    loss = loss_fn(
        prediction, torch.full((3,), -100.0), torch.zeros(3, dtype=torch.bool)
    )
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros(3))


@pytest.mark.parametrize("projector_type", ["linear", "xvector"])
def test_optional_reference_and_projector(projector_type):
    model = make_model(use_ref_audio=True, projector_type=projector_type)
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
@pytest.mark.parametrize("references", [False, True])
def test_frontend_wrapper_training_stats_and_checkpoint(multi_branch, references):
    from espnet2.asr.frontend.default import DefaultFrontend
    from espnet2.universa.espnet_model import ESPnetUniversaModel

    predictor = make_model(
        multi_branch=multi_branch,
        use_ref_audio=references,
        use_ref_text=references,
        vocab_size=5,
        text_encoder_params=dict(
            num_blocks=1, attention_heads=2, linear_units=16, input_layer="linear"
        ),
    )
    model = ESPnetUniversaModel(
        predictor, DefaultFrontend(n_fft=64, hop_length=16, n_mels=8)
    )
    audio, lengths = torch.randn(2, 256), torch.tensor([256, 200])
    extra = (
        dict(
            ref_audio=torch.randn(2, 256),
            ref_audio_lengths=lengths,
            ref_text=torch.tensor([[1, 2, 3], [2, -1, -1]]),
            ref_text_lengths=torch.tensor([3, 1]),
        )
        if references
        else {}
    )
    original_text = extra.get("ref_text", torch.empty(0)).clone()
    loss, stats, weight = model(
        audio, lengths, {"mos": torch.tensor([1.0, 2.0])}, **extra
    )
    assert torch.isfinite(loss) and weight == 2
    loss.backward()
    assert predictor.audio_encoder.embed[0].weight.grad is not None
    features = model.collect_feats(audio, lengths, **extra)
    assert features["audio"] is audio
    assert ("ref_audio" in features) == references
    torch.testing.assert_close(extra.get("ref_text", torch.empty(0)), original_text)
    model.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        result = model.inference(audio, lengths, **extra)
        absent = model.inference(audio, lengths)
    assert np.isfinite(result["mos"]).all() and np.isfinite(absent["mos"]).all()
    assert result["encoded_feat"].shape[-1] == (24 if references else 8)


@pytest.mark.parametrize("multi_branch", [False, True])
@pytest.mark.parametrize("weights", [{"mos": 0.0, "wer": 1.0}, {0: 0.0, 1: 1.0}])
def test_named_loss_weights(multi_branch, weights):
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
