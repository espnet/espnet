"""APIRunner: the infer stage driven by an Inference's declaration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.api.inference import Audio, Field, InferenceAPI
from espnet3.systems.base.api_runner import (
    APIRunner,
    declared_input_names,
)
from espnet3.systems.base.inference import infer
from espnet3.systems.base.inference_provider import InferenceProvider


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
            "segments": [{"text": "x", "start": 0.0, "end": speech.seconds}],
        }


class EchoProvider(InferenceProvider):
    @staticmethod
    def build_dataset(config):
        # arrays cannot live in an OmegaConf config; the test set is here
        return _items(config.dataset.size)

    @staticmethod
    def build_model(config):
        return Echo()


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


def test_forward_one_item_adds_the_id_and_copied_columns():
    data = _items()
    out = APIRunner.forward(1, dataset=data, model=Echo(), copy={"text": "ref"})
    assert list(out) == ["utt_id", "text", "echo", "segments", "ref"]
    assert out["utt_id"] == "utt1" and out["text"] == "200" and out["ref"] == "ref1"
    assert isinstance(out["echo"], Audio)


def test_forward_a_batch_goes_through_the_model_batch():
    data = _items()
    out = APIRunner.forward([0, 2], dataset=data, model=Echo())
    assert [o["utt_id"] for o in out] == ["utt0", "utt2"]
    assert [o["text"] for o in out] == ["100", "300"]


def test_forward_optional_inputs_and_the_index_as_id():
    data = [{"speech": np.zeros(8, dtype=np.float32), "prompt": "!"}]
    assert APIRunner.forward(0, dataset=data, model=Echo())["text"] == "8!"
    assert APIRunner.forward(0, dataset=data, model=Echo())["utt_id"] == "0"


def test_forward_says_what_is_missing_or_wrong():
    with pytest.raises(KeyError, match="has no 'speech', which Echo needs"):
        APIRunner.forward(0, dataset=[{"utt_id": "a"}], model=Echo())
    with pytest.raises(KeyError, match="has no 'text' to write as 'ref'"):
        APIRunner.forward(
            0, dataset=[{"speech": np.zeros(8)}], model=Echo(), copy={"text": "ref"}
        )
    with pytest.raises(TypeError, match="runs an Inference, not function"):
        APIRunner.forward(0, dataset=_items(), model=lambda speech: {"text": ""})
    with pytest.raises(TypeError, match="no call-time arguments"):
        APIRunner.forward(
            0, dataset=_items(), model=Echo(), model_kwargs={"beam_size": 3}
        )
    with pytest.raises(TypeError, match="applies no output_fn"):
        APIRunner.forward(0, dataset=_items(), model=Echo(), output_fn_path="src.x.f")
    with pytest.raises(KeyError, match="'text' is already an output"):
        APIRunner.forward(0, dataset=_items(), model=Echo(), copy={"text": "text"})


def test_write_record_writes_audio_as_wav_and_lists_as_json(tmp_path):
    writers = APIRunner.open_writers(tmp_path)
    result = APIRunner.forward(
        [0, 1], dataset=_items(), model=Echo(), copy={"text": "ref"}
    )
    APIRunner.write_record(writers, result, {}, idx_key="utt_id")
    APIRunner.close_writers(writers, {})
    assert (tmp_path / "text.scp").read_text() == "utt0 100\nutt1 200\n"
    assert (tmp_path / "ref.scp").read_text() == "utt0 ref0\nutt1 ref1\n"
    echo = (tmp_path / "echo.scp").read_text().splitlines()
    assert echo[0].startswith("utt0 ") and echo[0].endswith("echo/utt0.wav")
    import soundfile

    array, rate = soundfile.read(echo[0].split(" ", 1)[1])
    assert rate == 8000 and len(array) == 100
    segments = (tmp_path / "segments.scp").read_text().splitlines()[1]
    assert segments.endswith("segments/utt1.json")
    assert writers["artifact_configs"]["echo"] == {"type": "wav", "sample_rate": 8000}


def test_infer_runs_end_to_end_from_the_declaration(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test"}], "size": 5},
            "model": {"_target_": f"{__name__}.Echo"},
            "copy": {"text": "ref"},
            "batch_size": 2,
            "provider": {"_target_": f"{__name__}.EchoProvider"},
            "runner": {"_target_": "espnet3.systems.base.api_runner.APIRunner"},
            "parallel": {"env": "local", "n_workers": 1},
        }
    )
    infer(cfg)
    out = Path(tmp_path) / "test"
    assert (out / "text.scp").read_text().splitlines() == [
        f"utt{i} {100 * (i + 1)}" for i in range(5)
    ]
    assert (out / "ref.scp").read_text().splitlines() == [
        f"utt{i} ref{i}" for i in range(5)
    ]
    assert len((out / "echo.scp").read_text().splitlines()) == 5
    assert len((out / "segments.scp").read_text().splitlines()) == 5


def test_infer_still_wants_input_key_for_a_bare_model(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test"}], "size": 1},
            "model": {"_target_": "builtins.object"},
            "provider": {"_target_": f"{__name__}.EchoProvider"},
            "runner": {"_target_": "espnet3.systems.base.api_runner.APIRunner"},
            "parallel": {"env": "local", "n_workers": 1},
        }
    )
    with pytest.raises(RuntimeError, match="input_key must be set"):
        infer(cfg)
