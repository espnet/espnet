import pytest
import torch

from espnet3.utils.task_utils import get_espnet_model

TASK = "espnet3.systems.esp2_spk.task.SpeakerTask"
NUM_SPEAKERS = 8
FEAT_DIM = 40


@pytest.fixture(scope="module")
def model():
    """Build a small speaker model that reads precomputed features."""
    return get_espnet_model(
        TASK,
        dict(
            spk_num=NUM_SPEAKERS,
            frontend=None,
            input_size=FEAT_DIM,
            encoder="xvector",
            encoder_conf=dict(ndim=64, output_size=128),
            pooling="stats",
            projector="xvector",
            projector_conf=dict(output_size=32),
            loss="aamsoftmax",
        ),
    )


def test_task_registers_the_espnet_ssl_frontend():
    from espnet2.asr.frontend.espnet_ssl import ESPnetSSLFrontend
    from espnet3.systems.esp2_spk.task import frontend_choices

    assert frontend_choices.get_class("espnet_ssl") is ESPnetSSLFrontend


def test_training_batch_returns_a_classification_loss(model):
    speech = torch.randn(4, 100, FEAT_DIM)
    labels = torch.randint(0, NUM_SPEAKERS, (4, 1))

    loss, stats, weight = model(
        speech=speech,
        speech_lengths=torch.full((4,), 100),
        spk_labels=labels,
    )

    assert torch.isfinite(loss).all()
    assert set(stats) == {"loss", "accuracy"}
    assert int(weight) == 4
    assert model.trial_metric.update_count == 0


def test_classification_loss_runs_in_fp32_under_bf16_autocast(model):
    from espnet2.spk.espnet_model import ESPnetSpeakerModel

    torch.manual_seed(0)
    speech = torch.randn(4, 100, FEAT_DIM)
    lengths = torch.full((4,), 100)
    labels = torch.randint(0, NUM_SPEAKERS, (4, 1))
    was_training = model.training
    model.eval()  # fixed BatchNorm statistics, so both passes see one embedding
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _, _ = model(speech=speech, speech_lengths=lengths, spk_labels=labels)
            embd = ESPnetSpeakerModel.forward(
                model, speech=speech, speech_lengths=lengths, extract_embd=True
            )
        with torch.no_grad():
            expected, _, _ = model.loss(embd.float(), labels.squeeze())
    finally:
        model.train(was_training)

    assert loss.dtype == torch.float32
    # A bf16 head rounds the cosines to ~3 digits, which moves the loss by far
    # more than this tolerance.
    torch.testing.assert_close(loss.squeeze(), expected, rtol=1e-5, atol=1e-5)


def test_trial_batch_scores_pairs_instead_of_classifying(model):
    model.eval()
    enroll = torch.randn(3, 5, 100, FEAT_DIM)
    test = torch.randn(3, 5, 100, FEAT_DIM)
    labels = torch.tensor([[1], [0], [1]])

    loss, stats, weight = model(speech=enroll, speech2=test, spk_labels=labels)

    assert float(loss) == 0.0
    assert stats == {}
    assert int(weight) == 3

    scores, buffered_labels = model.trial_metric.compute()
    assert scores.shape == (3,)
    assert buffered_labels.tolist() == [1, 0, 1]
    model.trial_metric.reset()


def test_identical_utterances_score_one(model):
    model.eval()
    crops = torch.randn(2, 4, 100, FEAT_DIM)

    scores = model.score_trials(crops, crops.clone())

    assert torch.allclose(scores, torch.ones(2), atol=1e-3)
    model.trial_metric.reset()


def test_crop_embeddings_are_unit_norm(model):
    model.eval()

    embd = model.extract_crop_embeddings(torch.randn(2, 3, 100, FEAT_DIM))

    assert embd.shape[:2] == (2, 3)
    assert torch.allclose(embd.norm(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_trial_metric_is_not_saved_with_the_model(model):
    # ESPnet2 inference loads ``model.pth`` strictly, so the metric state must
    # stay out of the state dict.
    assert not any(key.startswith("trial_metric") for key in model.state_dict())


def test_trial_metric_on_an_empty_buffer(model):
    model.trial_metric.reset()

    with pytest.warns(UserWarning, match="before the ``update``"):
        scores, labels = model.trial_metric.compute()

    assert scores.numel() == 0 and labels.numel() == 0


def test_trial_batch_without_labels_is_rejected(model):
    model.eval()
    crops = torch.randn(2, 3, 100, FEAT_DIM)

    with pytest.raises(ValueError, match="spk_labels"):
        model(speech=crops, speech2=crops)

    model.trial_metric.reset()
