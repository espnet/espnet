import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from espnet3.api.inference.base import InferenceAPI
from espnet3.components.contract import Field
from espnet3.systems.base.inference_runner import InferenceRunner, _load_output_fn


class DummyProvider:
    pass


class DummyRunner(InferenceRunner):
    @staticmethod
    def forward(idx, *, dataset, model, **env):
        return {"idx": idx, "hyp": "h", "ref": "r"}


def test_resolve_idx_key_uses_default_utt_id():
    runner = DummyRunner(DummyProvider())

    assert runner.resolve_idx_key({"utt_id": "u1", "hyp": "h", "ref": "r"}) == "utt_id"


def test_resolve_idx_key_uses_explicit_configured_key():
    runner = DummyRunner(DummyProvider(), idx_key="sample_id")

    assert (
        runner.resolve_idx_key({"sample_id": "u1", "hyp": "h", "ref": "r"})
        == "sample_id"
    )


def test_resolve_idx_key_rejects_missing_configured_key():
    runner = DummyRunner(DummyProvider(), idx_key="sample_id")

    with pytest.raises(ValueError, match="idx_key='sample_id'"):
        runner.resolve_idx_key({"utt_id": "u1", "hyp": "h", "ref": "r"})


def test_load_output_fn_rejects_missing_module():
    with pytest.raises(ModuleNotFoundError):
        _load_output_fn("no.such.module.output_fn")


def test_forward_raises_without_input_key_kwarg():
    with pytest.raises(RuntimeError, match="input_key must be provided"):
        InferenceRunner.forward(0, dataset=[], model=lambda: None)


def test_forward_single_raises_key_error_for_missing_dataset_key():
    dataset = [{"speech": 1.0}]
    with pytest.raises(KeyError, match="Input key"):
        InferenceRunner.forward(
            0, dataset=dataset, model=lambda **kw: None, input_key="text"
        )


def test_forward_batched_raises_key_error_for_missing_dataset_key():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]
    with pytest.raises(KeyError, match="Input key"):
        InferenceRunner.forward(
            [0, 1], dataset=dataset, model=lambda **kw: None, input_key="text"
        )


def test_forward_batched_returns_model_output_without_output_fn():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]

    def model(speech):
        return {"result": speech}

    result = InferenceRunner.forward(
        [0, 1], dataset=dataset, model=model, input_key="speech"
    )
    assert result == {"result": [1.0, 2.0]}


def test_forward_single_returns_model_output_without_output_fn():
    dataset = [{"speech": 1.0}]

    def model(speech):
        return {"result": speech}

    result = InferenceRunner.forward(
        0, dataset=dataset, model=model, input_key="speech"
    )
    assert result == {"result": 1.0}


def test_forward_single_passes_model_kwargs_to_model():
    dataset = [{"speech": 1.0}]

    def model(speech, beam_size):
        return {"result": f"{speech}:{beam_size}"}

    result = InferenceRunner.forward(
        0,
        dataset=dataset,
        model=model,
        input_key="speech",
        model_kwargs={"beam_size": 2},
    )
    assert result == {"result": "1.0:2"}


def test_forward_batched_passes_model_kwargs_to_model():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]

    def model(speech, beam_size):
        return {"result": f"{speech}:{beam_size}"}

    result = InferenceRunner.forward(
        [0, 1],
        dataset=dataset,
        model=model,
        input_key="speech",
        model_kwargs={"beam_size": 4},
    )
    assert result == {"result": "[1.0, 2.0]:4"}


def test_forward_batched_wraps_model_exception_in_runtime_error():
    dataset = [{"speech": 1.0}]

    def failing_model(speech):
        raise ValueError("model broken")

    with pytest.raises(RuntimeError, match="one at a time") as info:
        InferenceRunner.forward(
            [0], dataset=dataset, model=failing_model, input_key="speech"
        )
    assert isinstance(info.value.__cause__, ValueError)


def test_forward_batched_reports_out_of_memory_with_lengths_and_batch_size():
    """An OOM is not "your model does not support batches": name the items."""
    import numpy as np
    import torch

    dataset = [{"speech": np.zeros(16000)}, {"speech": np.zeros(8000)}]

    def model(speech):
        raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

    with pytest.raises(RuntimeError, match="ran out of memory") as info:
        InferenceRunner.forward(
            [0, 1], dataset=dataset, model=model, input_key="speech"
        )
    message = str(info.value)
    assert "[16000, 8000]" in message
    assert "batch_size" in message and "2 items" in message
    assert "set batch_size to None" not in message
    assert isinstance(info.value.__cause__, torch.OutOfMemoryError)


# ---------------------------------------------------------------------------
# merge(): fields.json
# ---------------------------------------------------------------------------


class _FakeInferenceModel(InferenceAPI):
    """A minimal, never-instantiated InferenceAPI subclass, for its
    declared `outputs` only (`_model_output_kind` reads the class, not an
    instance)."""

    inputs = (Field("speech", "audio"),)
    outputs = (Field("text", "text"),)

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
        raise NotImplementedError

    def run(self, **inputs):
        raise NotImplementedError


def _write_shard(shard_dir: Path, field_keys: list[str], rows: dict[str, str]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "field_keys.txt").write_text(
        "\n".join(field_keys) + "\n", encoding="utf-8"
    )
    for key in field_keys:
        (shard_dir / f"{key}.scp").write_text(f"utt1 {rows[key]}\n", encoding="utf-8")


def _make_merge_runner(tmp_path: Path, *, params=None, config=None) -> DummyRunner:
    provider = SimpleNamespace(params=params or {}, config=config)
    return DummyRunner(provider, output_dir=tmp_path)


def test_merge_writes_fields_json_kind_from_output_artifact(tmp_path: Path):
    shard_dir = tmp_path / "split.0"
    _write_shard(shard_dir, ["hyp_audio"], {"hyp_audio": "/tmp/out.wav"})
    runner = _make_merge_runner(
        tmp_path, params={"output_artifacts": {"hyp_audio": {"type": "wav"}}}
    )

    runner.merge([shard_dir])

    payload = json.loads((tmp_path / "fields.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["idx_key"] == "utt_id"
    assert payload["fields"]["hyp_audio"] == {"kind": "audio", "artifact": "wav"}


def test_merge_writes_fields_json_kind_from_model_outputs(tmp_path: Path):
    shard_dir = tmp_path / "split.0"
    _write_shard(shard_dir, ["text"], {"text": "hello"})
    model_target = f"{__name__}._FakeInferenceModel"
    runner = _make_merge_runner(
        tmp_path, config=SimpleNamespace(model=SimpleNamespace(_target_=model_target))
    )

    runner.merge([shard_dir])

    payload = json.loads((tmp_path / "fields.json").read_text())
    assert payload["fields"]["text"] == {"kind": "text"}


def test_merge_writes_fields_json_kind_null_when_unknown(tmp_path: Path):
    shard_dir = tmp_path / "split.0"
    _write_shard(shard_dir, ["ref"], {"ref": "the reference text"})
    runner = _make_merge_runner(tmp_path)

    runner.merge([shard_dir])

    payload = json.loads((tmp_path / "fields.json").read_text())
    assert payload["fields"]["ref"] == {"kind": None}
