from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from espnet3.systems.spk.inference import ESPnet2Speech2Score
from espnet3.utils.task_utils import get_espnet_model, save_espnet_config

TASK = "espnet3.systems.spk.task.SpeakerTask"
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
