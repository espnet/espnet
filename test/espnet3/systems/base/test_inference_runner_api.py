"""InferenceRunner with an Inference: the infer stage driven by the declaration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.api.inference import Audio, Field, InferenceAPI
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.base.inference import infer
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner, declared_input_names
from espnet3.systems.base.metric import measure

RUNNER = "espnet3.systems.base.inference_runner.InferenceRunner"


class Echo(InferenceAPI):
    """Text out of audio, with an optional prompt and an audio echo."""

    inputs = (Field("speech", "audio"), Field("prompt", "text", optional=True))
    outputs = (Field("text", "text"), Field("echo", "audio"))

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
        return cls()

    sample_rate = 8000

    def run(self, speech, prompt=""):
        return {
            "text": f"{len(speech.array)}{prompt}",
            "echo": speech,
            "segments": [{"text": "x", "start": 0.0, "end": speech.duration}],
        }


class EchoProvider(InferenceProvider):
    @staticmethod
    def build_dataset(config):
        # arrays cannot live in an OmegaConf config; the test set is here
        return _items(config.dataset.size)

    @staticmethod
    def build_model(config):
        return Echo()


class Match(BaseMetric):
    """Pairs the data's `text` with the model's `text`, by alias."""

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("n", "number"),)

    ref_key = "dataset:text"
    hyp_key = "text"

    def input_sources(self):
        return {"ref": self.ref_key, "hyp": self.hyp_key}

    def __call__(self, data, test_name, inference_dir):
        rows = [row for _, row in self.iter_inputs(data, "ref", "hyp")]
        return {"n": len(rows), "refs": [r["ref"] for r in rows]}


class MatchAudio(Match):
    """Match, but pairing the data's `speech` with the model's audio `echo`."""

    inputs = (Field("ref", "text"), Field("hyp", "audio"))

    ref_key = "dataset:speech"
    hyp_key = "echo"


def _items(n=3):
    return [
        {
            "utt_id": f"utt{i}",
            "speech": np.zeros(100 * (i + 1), dtype=np.float32),
            "text": f"ref{i}",
        }
        for i in range(n)
    ]


def test_declared_input_names_reads_the_class_without_building_it():
    assert declared_input_names(
        OmegaConf.create({"model": {"_target_": f"{__name__}.Echo"}})
    ) == ["speech", "prompt"]
    assert (
        declared_input_names(
            OmegaConf.create({"model": {"_target_": "builtins.object"}})
        )
        is None
    )
    assert (
        declared_input_names(OmegaConf.create({"model": {"_target_": "no.such.Thing"}}))
        is None
    )
    assert declared_input_names(OmegaConf.create({"model": None})) is None


def test_forward_one_item_adds_the_id_and_nothing_else():
    data = _items()
    out = InferenceRunner.forward(1, dataset=data, model=Echo())
    assert list(out) == ["utt_id", "text", "echo", "segments"]
    assert out["utt_id"] == "utt1" and out["text"] == "200"
    assert isinstance(out["echo"], Audio)


def test_forward_a_batch_goes_through_the_model_batch():
    data = _items()
    out = InferenceRunner.forward([0, 2], dataset=data, model=Echo())
    assert [o["utt_id"] for o in out] == ["utt0", "utt2"]
    assert [o["text"] for o in out] == ["100", "300"]


def test_forward_optional_inputs_and_the_index_as_id():
    data = [{"speech": np.zeros(8, dtype=np.float32), "prompt": "!"}]
    assert InferenceRunner.forward(0, dataset=data, model=Echo())["text"] == "8!"
    assert InferenceRunner.forward(0, dataset=data, model=Echo())["utt_id"] == "0"


def test_forward_says_what_is_missing_or_wrong():
    with pytest.raises(KeyError, match="has no 'speech', which Echo needs"):
        InferenceRunner.forward(0, dataset=[{"utt_id": "a"}], model=Echo())
    with pytest.raises(RuntimeError, match="input_key must be provided"):
        InferenceRunner.forward(0, dataset=_items(), model=lambda speech: {"text": ""})
    with pytest.raises(TypeError, match="no call-time arguments"):
        InferenceRunner.forward(
            0, dataset=_items(), model=Echo(), model_kwargs={"beam_size": 3}
        )
    with pytest.raises(TypeError, match="applies no output_fn"):
        InferenceRunner.forward(
            0, dataset=_items(), model=Echo(), output_fn_path="src.x.f"
        )


def test_write_record_writes_audio_as_wav_and_lists_as_json(tmp_path):
    writers = InferenceRunner.open_writers(tmp_path)
    result = InferenceRunner.forward([0, 1], dataset=_items(), model=Echo())
    InferenceRunner.write_record(writers, result, {}, idx_key="utt_id")
    InferenceRunner.close_writers(writers, {})
    assert (tmp_path / "text.scp").read_text() == "utt0 100\nutt1 200\n"
    assert not (tmp_path / "ref.scp").exists()
    echo = (tmp_path / "echo.scp").read_text().splitlines()
    assert echo[0].startswith("utt0 ") and echo[0].endswith("echo/utt0.wav")
    import soundfile

    array, rate = soundfile.read(echo[0].split(" ", 1)[1])
    assert rate == 8000 and len(array) == 100
    segments = (tmp_path / "segments.scp").read_text().splitlines()[1]
    assert segments.endswith("segments/utt1.json")
    assert "echo" not in writers["artifact_configs"]  # per record, not shared


def test_write_record_writes_each_audio_at_its_own_rate(tmp_path):
    import soundfile

    writers = InferenceRunner.open_writers(tmp_path)
    records = [
        {"utt_id": "a", "echo": Audio(np.zeros(80, dtype=np.float32), 8000)},
        {"utt_id": "b", "echo": Audio(np.zeros(160, dtype=np.float32), 16000)},
    ]
    InferenceRunner.write_record(writers, records, {}, idx_key="utt_id")
    InferenceRunner.close_writers(writers, {})
    rates = [
        soundfile.read(line.split(" ", 1)[1])[1]
        for line in (tmp_path / "echo.scp").read_text().splitlines()
    ]
    assert rates == [8000, 16000]


def test_write_record_writes_multichannel_audio_channels_last(tmp_path):
    import soundfile

    class Stereo(Echo):
        inputs = (Field("speech", "audio", channels=None),)
        outputs = (Field("text", "text"), Field("echo", "audio", channels=None))

        def run(self, speech, prompt=""):
            return {"text": str(speech.channels), "echo": speech}

    data = [{"utt_id": "s", "speech": np.zeros((2, 100), dtype=np.float32)}]
    writers = InferenceRunner.open_writers(tmp_path)
    InferenceRunner.write_record(
        writers, InferenceRunner.forward(0, dataset=data, model=Stereo()), {}
    )
    InferenceRunner.close_writers(writers, {})
    path = (tmp_path / "echo.scp").read_text().split(" ", 1)[1].strip()
    array, rate = soundfile.read(path)
    assert array.shape == (100, 2) and rate == 8000
    assert (tmp_path / "text.scp").read_text() == "s 2\n"


def test_infer_runs_end_to_end_from_the_declaration(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test"}], "size": 5},
            "model": {"_target_": f"{__name__}.Echo"},
            "batch_size": 2,
            "provider": {"_target_": f"{__name__}.EchoProvider"},
            "runner": {"_target_": RUNNER},
            "parallel": {"env": "local", "n_workers": 1},
        }
    )
    infer(cfg)
    out = Path(tmp_path) / "test"
    assert (out / "text.scp").read_text().splitlines() == [
        f"utt{i} {100 * (i + 1)}" for i in range(5)
    ]
    assert not (out / "ref.scp").exists()
    assert len((out / "echo.scp").read_text().splitlines()) == 5
    assert len((out / "segments.scp").read_text().splitlines()) == 5


@pytest.mark.parametrize("utt_id", ["../escape", "a/b", "", "..", "utt 1", "utt\\n1"])
def test_write_record_refuses_an_id_that_is_a_path_or_breaks_a_line(tmp_path, utt_id):
    writers = InferenceRunner.open_writers(tmp_path)
    data = [{"utt_id": utt_id, "speech": np.zeros(8, dtype=np.float32)}]
    result = InferenceRunner.forward(0, dataset=data, model=Echo())
    with pytest.raises(ValueError, match="plain token"):
        InferenceRunner.write_record(writers, result, {}, idx_key="utt_id")
    InferenceRunner.close_writers(writers, {})
    # nothing was written: no line, no artifact, nowhere
    assert not (tmp_path.parent / "escape.wav").exists()
    assert not (tmp_path / "text.scp").exists()
    assert not (tmp_path / "echo").exists()


def test_measure_reads_the_reference_from_the_test_set(tmp_path):
    """`ref_key: dataset:text` scores against the data, not a copied file."""
    inference_cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test"}], "size": 3},
            "model": {"_target_": f"{__name__}.Echo"},
            "provider": {"_target_": f"{__name__}.EchoProvider"},
            "runner": {"_target_": RUNNER},
            "parallel": {"env": "local", "n_workers": 1},
        }
    )
    infer(inference_cfg)
    metrics_cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test"}]},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.Match"},
                    "inputs": {"ref": "dataset:text", "hyp": "text"},
                }
            ],
        }
    )
    results = measure(metrics_cfg, inference_config=inference_cfg)
    (result,) = results.values()
    assert result["test"] == {"n": 3, "refs": ["ref0", "ref1", "ref2"]}
    written = tmp_path / "test" / "dataset" / "text.scp"
    assert written.read_text().splitlines() == ["utt0 ref0", "utt1 ref1", "utt2 ref2"]
    # a second run reuses the file rather than reading the set again
    written.write_text("utt0 edited\nutt1 ref1\nutt2 ref2\n")
    (result,) = measure(metrics_cfg, inference_config=inference_cfg).values()
    assert result["test"]["refs"][0] == "edited"
    # a column the set does not have is named, with what it does have
    missing = OmegaConf.create(OmegaConf.to_container(metrics_cfg))
    missing.metrics[0].inputs = {"ref": "dataset:nope", "hyp": "text"}
    with pytest.raises(KeyError, match="has no 'nope'; it has \\['speech', 'text'"):
        measure(missing, inference_config=inference_cfg)
    # the failed run left nothing a later run would take as finished
    assert not (tmp_path / "test" / "dataset" / "nope.scp").exists()
    # without the inference config the model declares no outputs either,
    # so the contract check rejects it before the dataset read is reached
    written.unlink()
    with pytest.raises(ValueError, match="declares no outputs"):
        measure(metrics_cfg)


def test_measure_writes_a_dataset_waveform_as_an_artifact(tmp_path):
    """A `dataset:` column that is not a scalar lands beside the .scp, as told."""
    import shutil

    import soundfile

    inference_cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test"}], "size": 2},
            "model": {"_target_": f"{__name__}.Echo"},
            "provider": {"_target_": f"{__name__}.EchoProvider"},
            "runner": {"_target_": RUNNER},
            "parallel": {"env": "local", "n_workers": 1},
        }
    )
    infer(inference_cfg)
    metrics_cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test"}]},
            "dataset_artifacts": {"speech": {"type": "wav", "sample_rate": 8000}},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.MatchAudio"},
                    "inputs": {"ref": "dataset:speech", "hyp": "echo"},
                }
            ],
        }
    )
    (result,) = measure(metrics_cfg, inference_config=inference_cfg).values()
    assert result["test"]["n"] == 2
    ref0 = Path(result["test"]["refs"][0])
    assert ref0 == tmp_path / "test" / "dataset" / "speech" / "utt0.wav"
    array, rate = soundfile.read(ref0)
    assert array.shape == (100,) and rate == 8000
    # with nothing said, an array is .npy
    plain_dir = tmp_path / "plain"
    (plain_dir / "test").mkdir(parents=True)
    shutil.copy(tmp_path / "test" / "echo.scp", plain_dir / "test")
    plain = OmegaConf.create(
        {
            "inference_dir": str(plain_dir),
            "dataset": {"test": [{"name": "test"}]},
            "metrics": metrics_cfg.metrics,
        }
    )
    (result,) = measure(plain, inference_config=inference_cfg).values()
    assert result["test"]["refs"][0].endswith("dataset/speech/utt0.npy")


def test_infer_still_wants_input_key_for_a_bare_model(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test"}], "size": 1},
            "model": {"_target_": "builtins.object"},
            "provider": {"_target_": f"{__name__}.EchoProvider"},
            "runner": {"_target_": RUNNER},
            "parallel": {"env": "local", "n_workers": 1},
        }
    )
    with pytest.raises(RuntimeError, match="input_key must be set"):
        infer(cfg)


class Silent(Echo):
    """No speech found: the segments output is an empty list."""

    def run(self, speech, prompt=""):
        return {"text": "", "echo": speech, "segments": []}


def test_write_record_writes_an_empty_segments_list(tmp_path):
    """A silent utterance's `[]` is a JSON document, not an unsupported list."""
    writers = InferenceRunner.open_writers(tmp_path)
    result = InferenceRunner.forward(0, dataset=_items(1), model=Silent())
    InferenceRunner.write_record(writers, result, {}, idx_key="utt_id")
    InferenceRunner.close_writers(writers, {})
    path = (tmp_path / "segments.scp").read_text().split(" ", 1)[1].strip()
    assert json.loads(Path(path).read_text()) == {"segments": []}
