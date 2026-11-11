from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from espnet3.contract.inference import load_inference_contract
from espnet3.systems.esp2_spk.src.inference_runner import ESP2SPKInferenceRunner
from espnet3.systems.esp2_spk.src.speech2score import (
    ESPnet2Speech2Score,
    TrialScorer,
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


class _CropRecordingModel:
    """Stands in for the speaker model: records the crops it is scored on.

    The score is the cosine of the two utterances' mean crops, so it depends
    on what was cropped and a double crop would change it.
    """

    def __init__(self):
        self.calls = []

    def score_trials(self, speech, speech2):
        self.calls.append((tuple(speech.shape), tuple(speech2.shape)))
        a, b = speech.mean(dim=1), speech2.mean(dim=1)
        return torch.nn.functional.cosine_similarity(a, b)


def _trial_scorer(target_duration=0.5, num_eval=4, sample_rate=16000):
    """A ``TrialScorer`` around the recording model, without a checkpoint."""
    backend = ESPnet2Speech2Score.__new__(ESPnet2Speech2Score)
    backend.model = _CropRecordingModel()
    backend.device = "cpu"
    backend.dtype = "float32"
    backend.sample_rate = sample_rate
    backend.crop_samples = int(target_duration * sample_rate)
    backend.num_eval = num_eval
    scorer = TrialScorer.__new__(TrialScorer)
    scorer.backend = backend
    scorer.sample_rate = sample_rate
    return scorer


CONTRACT = load_inference_contract(
    {
        "inputs": {
            name: {
                "kind": "audio",
                "sample_rate": 16000,
                "channels": 1,
                "normalize": ["first_channel"],
            }
            for name in ("speech", "speech2")
        },
        "outputs": {"score": {"kind": "number"}},
    }
)


def _run(dataset, model, idx=0):
    return ESP2SPKInferenceRunner.forward(
        idx,
        dataset=dataset,
        model=model,
        input_key=["speech", "speech2"],
        inference_contract=CONTRACT,
    )


def test_trial_scorer_crops_each_whole_waveform_once():
    scorer = _trial_scorer()
    rng = np.random.default_rng(0)
    wav, wav2 = rng.standard_normal(16000), rng.standard_normal(12000)

    scorer(wav, wav2)

    assert scorer.backend.model.calls == [((1, 4, 8000), (1, 4, 8000))]


def test_trial_scorer_wraps_a_clip_shorter_than_one_crop():
    scorer = _trial_scorer()
    short = np.random.default_rng(1).standard_normal(3000)

    score = scorer(short, short)

    assert scorer.backend.model.calls == [((1, 4, 8000), (1, 4, 8000))]
    assert score == pytest.approx(1.0, abs=1e-5)


def test_runner_scores_a_raw_trial_like_the_scorer_itself():
    scorer = _trial_scorer()
    rng = np.random.default_rng(2)
    wav = rng.standard_normal(16000).astype("float32")
    wav2 = rng.standard_normal(9000).astype("float32")
    dataset = [{"speech": wav, "speech2": wav2, "spk_labels": "1"}]

    out = _run(dataset, scorer)

    assert out == {"score": pytest.approx(scorer(wav, wav2)), "utt_id": "0"}
    assert set(out) == {"score", "utt_id"}


def test_runner_resamples_and_takes_the_first_channel():
    scorer = _trial_scorer()
    stereo_8k = np.zeros((2, 8000), dtype="float32")
    dataset = [{"speech": (8000, stereo_8k), "speech2": np.zeros(16000)}]

    _run(dataset, scorer)

    # 1 s at 8 kHz reaches the scorer as 16,000 mono samples, then 4 crops.
    assert scorer.backend.model.calls == [((1, 4, 8000), (1, 4, 8000))]


def test_runner_refuses_a_batch():
    wav = np.zeros(16000, dtype="float32")
    dataset = [{"speech": wav, "speech2": wav}] * 2

    with pytest.raises(ValueError, match="batch_size: null"):
        _run(dataset, _trial_scorer(), idx=[0, 1])


def test_runner_rejects_non_finite_audio():
    wav = np.zeros(16000, dtype="float32")
    bad = wav.copy()
    bad[10] = np.nan

    with pytest.raises(Exception):
        _run([{"speech": wav, "speech2": bad}], _trial_scorer())
