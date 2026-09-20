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
