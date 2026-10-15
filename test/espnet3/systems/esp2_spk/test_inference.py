from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from espnet3.systems.esp2_spk.inference import (
    ESPnet2Speech2Score,
    Inference,
    eval_crops,
)
from espnet3.utils.task_utils import get_espnet_model, save_espnet_config

TASK = "espnet3.systems.esp2_spk.task.SpeakerTask"
FEAT_DIM = 40
MODEL_CONF = dict(
    spk_num=8,
    frontend=None,
    input_size=FEAT_DIM,
    encoder="xvector",
    encoder_conf=dict(ndim=64, output_size=128),
    pooling="stats",
    projector="xvector",
    projector_conf=dict(output_size=32),
    loss="aamsoftmax",
)


@pytest.fixture(scope="module")
def scorer(tmp_path_factory) -> ESPnet2Speech2Score:
    """Train-free round trip: save a config, save weights, load a scorer."""
    exp_dir = tmp_path_factory.mktemp("exp")
    save_espnet_config(TASK, OmegaConf.create({"model": MODEL_CONF}), str(exp_dir))
    model = get_espnet_model(TASK, MODEL_CONF)
    torch.save(model.state_dict(), exp_dir / "model.pth")
    return ESPnet2Speech2Score(
        train_config=str(exp_dir / "config.yaml"),
        model_file=str(exp_dir / "model.pth"),
    )


def test_scoring_a_trial_returns_a_float(scorer):
    crops = np.random.randn(5, 100, FEAT_DIM).astype("float32")

    score = scorer(crops, np.random.randn(5, 100, FEAT_DIM).astype("float32"))

    assert isinstance(score, float)
    assert -1.001 <= score <= 1.001


def test_identical_utterances_score_one(scorer):
    crops = np.random.randn(5, 100, FEAT_DIM).astype("float32")

    assert scorer(crops, crops) == pytest.approx(1.0, abs=1e-3)


def test_torch_inputs_are_accepted(scorer):
    crops = torch.randn(3, 100, FEAT_DIM)

    assert scorer(crops, crops) == pytest.approx(1.0, abs=1e-3)


def test_embedding_extraction_returns_one_vector_per_utterance(scorer):
    embedding = scorer.extract_embedding(
        np.random.randn(5, 100, FEAT_DIM).astype("float32")
    )

    assert embedding.shape == (32,)


def test_missing_checkpoint_is_reported(tmp_path: Path):
    save_espnet_config(TASK, OmegaConf.create({"model": MODEL_CONF}), str(tmp_path))

    with pytest.raises(Exception):
        ESPnet2Speech2Score(
            train_config=str(tmp_path / "config.yaml"),
            model_file=str(tmp_path / "missing.pth"),
        )


def test_eval_crops_are_evenly_spaced_and_cover_both_ends():
    crops = eval_crops(np.arange(10.0), crop_samples=4, num_eval=3)

    assert crops.shape == (3, 4)
    np.testing.assert_array_equal(crops[:, 0], [0.0, 3.0, 6.0])
    assert crops[-1, -1] == 9.0


def test_eval_crops_wrap_an_utterance_shorter_than_one_crop():
    crops = eval_crops(np.arange(3.0), crop_samples=5, num_eval=2)

    assert crops.shape == (2, 5)
    np.testing.assert_array_equal(crops[0], [0.0, 1.0, 2.0, 0.0, 1.0])


class _FakeScorer:
    """Stands in for ESPnet2Speech2Score: records what `Inference` hands it."""

    sample_rate = 16000

    def __init__(self):
        self.calls = []

    def score(self, speech, speech2):
        self.calls.append((speech, speech2))
        return 0.5


def test_inference_declares_a_trial_in_and_a_score_out():
    assert [f.name for f in Inference.inputs] == ["speech", "speech2"]
    assert [(f.name, f.kind) for f in Inference.outputs] == [("score", "number")]


def test_inference_hands_the_scorer_two_mono_waveforms():
    backend = _FakeScorer()
    model = Inference(backend)
    wav = np.random.randn(16000).astype("float32")

    assert model(wav, (16000, wav[:8000])) == {"score": 0.5}
    speech, speech2 = backend.calls[0]
    assert speech.shape == (16000,) and speech2.shape == (8000,)


def test_inference_resamples_to_the_model_rate():
    backend = _FakeScorer()
    Inference(backend)((8000, np.zeros(8000, dtype="float32")), np.zeros(16000))

    assert backend.calls[0][0].shape == (16000,)
