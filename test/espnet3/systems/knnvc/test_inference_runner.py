"""The kNN-VC runner behind the shared inference contract."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import egs3.TEMPLATE.knnvc.src.contracts  # noqa: F401  registers reference_audio
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.metric import measure
from espnet3.systems.knnvc.metrics.intelligibility import ASRIntelligibility
from espnet3.systems.knnvc.src.inference_runner import KNNVCInferenceRunner

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_prepare_model_inputs_unwraps_audio     | Audio -> arrays, one per     |
# |                          | reference; bare arrays pass through unchanged.  |
# | test_batches_are_refused                    | A list of sources raises     |
# |                                             | before the model runs.       |
# | test_forward_checks_both_sides_of_the_boundary | Rates are resampled,      |
# |                          | pair_id is attached, wav comes back as 16k Audio.|
# | test_forward_rejects_bad_inputs             | Stereo, NaN, an empty        |
# |                          | reference list and a missing key are refused.   |
# | test_measure_scores_wav_against_the_dataset_text | measure() binds        |
# |                          | ref to dataset:text keyed by pair_id.           |

SR = 16000
CONTRACT = {
    "imports": ["egs3.TEMPLATE.knnvc.src.contracts"],
    "inputs": {
        "speech": {"kind": "audio", "sample_rate": SR, "channels": 1},
        "reference_speech": {
            "kind": "reference_audio",
            "sample_rate": SR,
            "channels": 1,
        },
        "target_speaker": {"kind": "text"},
    },
    "outputs": {"wav": {"kind": "audio", "sample_rate": SR, "channels": 1}},
}
INPUT_KEY = ["speech", "reference_speech", "target_speaker"]


class FakeKNNVC:
    """Record what the runner hands over and return a waveform of the same length."""

    sample_rate = SR

    def __init__(self):
        self.calls = []

    def __call__(self, speech, reference_speech, target_speaker=None):
        self.calls.append((speech, reference_speech, target_speaker))
        return np.zeros_like(speech, dtype=np.float32) + 0.1


def _wav(seconds, seed=0, rate=SR):
    rng = np.random.RandomState(seed)
    return (rng.randn(int(seconds * rate)) * 0.1).astype(np.float32)


def _item(pair_id="u1_to_spkB", **overrides):
    item = {
        "speech": _wav(0.5),
        "reference_speech": [_wav(0.3, 1), _wav(0.4, 2)],
        "target_speaker": "spkB",
        "pair_id": pair_id,
        "text": "HELLO",
    }
    item.update(overrides)
    return item


def test_prepare_model_inputs_unwraps_audio():
    source = Audio(_wav(0.5), SR)
    refs = [Audio(_wav(0.3, 1), SR), Audio(_wav(0.4, 2), SR)]
    out = KNNVCInferenceRunner.prepare_model_inputs(
        {"speech": source, "reference_speech": refs, "target_speaker": "spk"}
    )
    assert out["speech"] is source.array
    assert [r.shape for r in out["reference_speech"]] == [r.array.shape for r in refs]
    assert out["target_speaker"] == "spk"

    # Without a contract (a raw call), arrays go through untouched.
    raw = {"speech": _wav(0.5), "reference_speech": [_wav(0.3)]}
    assert KNNVCInferenceRunner.prepare_model_inputs(raw)["speech"] is raw["speech"]


def test_batches_are_refused():
    with pytest.raises(ValueError, match="batch_size: null"):
        KNNVCInferenceRunner.prepare_model_inputs(
            {"speech": [_wav(0.5), _wav(0.5)], "reference_speech": [[_wav(0.3)]]}
        )
    model = FakeKNNVC()
    with pytest.raises(ValueError, match="batch_size: null"):
        KNNVCInferenceRunner.forward(
            [0, 1],
            dataset=[_item(), _item("u2_to_spkB")],
            model=model,
            input_key=INPUT_KEY,
            inference_contract=CONTRACT,
        )
    assert model.calls == []


def test_forward_checks_both_sides_of_the_boundary():
    model = FakeKNNVC()
    # The source carries a rate and is resampled; the references come as a
    # rate-less list (16 kHz by declaration) and a (rate, samples) pair.
    item = _item(
        speech=(8000, _wav(0.5, rate=8000)),
        reference_speech=[_wav(0.3, 1), (SR, _wav(0.4, 2))],
    )

    result = KNNVCInferenceRunner.forward(
        0,
        dataset=[item],
        model=model,
        input_key=INPUT_KEY,
        inference_contract=CONTRACT,
        idx_key="pair_id",
    )

    ((speech, references, speaker),) = model.calls
    assert isinstance(speech, np.ndarray) and speech.shape == (int(0.5 * SR),)
    assert [len(r) for r in references] == [int(0.3 * SR), int(0.4 * SR)]
    assert speaker == "spkB"
    assert set(result) == {"wav", "pair_id"}
    assert result["pair_id"] == "u1_to_spkB"
    assert isinstance(result["wav"], Audio)
    assert result["wav"].rate == SR and result["wav"].channels == 1
    assert result["wav"].array.shape == speech.shape


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"speech": np.zeros((2, SR), dtype=np.float32)}, "channel"),
        ({"speech": np.full(SR, np.nan, dtype=np.float32)}, "finite"),
        ({"reference_speech": []}, "at least one"),
        ({"reference_speech": _wav(0.3)}, "list of reference"),
        ({"reference_speech": [np.zeros((2, SR), dtype=np.float32)]}, "channel"),
        ({"target_speaker": 7}, "must be str"),
    ],
)
def test_forward_rejects_bad_inputs(overrides, error):
    model = FakeKNNVC()
    with pytest.raises((TypeError, ValueError), match=error):
        KNNVCInferenceRunner.forward(
            0,
            dataset=[_item(**overrides)],
            model=model,
            input_key=INPUT_KEY,
            inference_contract=CONTRACT,
        )
    assert model.calls == []

    item = _item()
    del item["reference_speech"]
    with pytest.raises(KeyError, match="reference_speech"):
        KNNVCInferenceRunner.forward(
            0,
            dataset=[item],
            model=model,
            input_key=INPUT_KEY,
            inference_contract=CONTRACT,
        )


class ConversionProvider(InferenceProvider):
    """Serve two conversion items without a corpus."""

    @staticmethod
    def build_dataset(config):
        return [
            _item("u1_to_spkB", text="THE CAT SAT"),
            _item("u2_to_spkB", text="A DOG RAN"),
        ]


class FakeASR:
    def __init__(self, hypotheses):
        self.hypotheses = list(hypotheses)

    def __call__(self, speech):
        return [(self.hypotheses.pop(0), None, None, None)]


def test_measure_scores_wav_against_the_dataset_text(tmp_path: Path):
    inference_dir = tmp_path / "inference"
    test_dir = inference_dir / "test"
    (test_dir / "wav").mkdir(parents=True)
    import soundfile as sf

    lines = []
    for pair_id in ("u1_to_spkB", "u2_to_spkB"):
        path = test_dir / "wav" / f"{pair_id}.wav"
        sf.write(path, _wav(0.5), SR)
        lines.append(f"{pair_id} {path}")
    (test_dir / "wav.scp").write_text("\n".join(lines) + "\n")

    inference_config = OmegaConf.create(
        {
            "idx_key": "pair_id",
            "dataset": {"test": [{"name": "test"}]},
            "provider": {
                "_target_": f"{ConversionProvider.__module__}.ConversionProvider"
            },
        }
    )
    metrics_config = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "metrics": [
                {
                    "metric": {
                        "_target_": (
                            f"{ASRIntelligibility.__module__}.ASRIntelligibility"
                        ),
                        "asr": {
                            "_target_": f"{FakeASR.__module__}.FakeASR",
                            "hypotheses": ["THE CAT SAT", "A DOG WALKED"],
                        },
                    },
                    "inputs": {"wav": "wav", "ref": "dataset:text"},
                }
            ],
        }
    )

    results = measure(metrics_config, inference_config=inference_config)

    (scores,) = results.values()
    assert scores["test"]["ASR_WER"] == pytest.approx(100 / 6, abs=0.01)
    # The transcript was read from the data, keyed by the conversion's pair_id.
    written = (test_dir / "dataset" / "text.scp").read_text().splitlines()
    assert written == ["u1_to_spkB THE CAT SAT", "u2_to_spkB A DOG RAN"]
    assert (test_dir / "asr_wer_alignment").is_file()
