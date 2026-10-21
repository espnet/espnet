"""Exercise shared ASR contracts through recipe, publication and demo."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from espnet3.publication import InferenceModel
from espnet3.publication.demo.assets import DEFAULT_UI_ASSETS
from espnet3.publication.demo.session import DemoSession
from espnet3.systems.base.inference import infer
from espnet3.systems.base.inference_provider import InferenceProvider


class RecordingASR:
    """Native-shaped, input-dependent hypotheses with observable batch calls."""

    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, speech):
        self.calls.append(speech)
        if self.fail:
            raise RuntimeError("backend failed")

        def one(wave):
            return [(f"  WORD-{int(np.sum(wave))}!  ", [], [], None)]

        return (
            [one(wave) for wave in speech] if isinstance(speech, list) else one(speech)
        )


@pytest.fixture
def shared_paths(tmp_path, monkeypatch):
    """Create a real packed config and wire only backend/dataset construction."""
    backend = RecordingASR()
    import espnet3.parallel.parallel as parallel_module

    monkeypatch.setattr(parallel_module, "parallel_config", None)
    samples = []
    config = OmegaConf.create(
        {
            "model": {},
            "parallel": {
                "env": "local",
                "n_workers": 1,
                "options": {"processes": False, "dashboard_address": None},
            },
            "inference_dir": str(tmp_path / "inference"),
            "dataset": {"test": [{"name": "test"}]},
            "input_key": "speech",
            "output_keys": ["text"],
            "batch_size": 2,
            "provider": {
                "_target_": "espnet3.systems.base.inference_provider.InferenceProvider"
            },
            "runner": {
                "_target_": "espnet3.systems.esp2_asr.inference.ASRInferenceRunner"
            },
            "contract": {
                "inputs": {
                    "speech": {
                        "kind": "audio",
                        "sample_rate": 16000,
                        "rate_policy": "reject",
                        "channels": 1,
                    }
                },
                "outputs": {
                    "text": {"kind": "text", "normalize": ["strip", "to_lower"]}
                },
            },
        }
    )
    monkeypatch.setattr(
        InferenceProvider, "build_model", staticmethod(lambda config: backend)
    )
    monkeypatch.setattr(
        InferenceProvider, "build_dataset", staticmethod(lambda config: samples)
    )
    pack = tmp_path / "pack"
    (pack / "conf").mkdir(parents=True)
    OmegaConf.save(config, pack / "conf" / "inference.yaml")
    OmegaConf.save(
        OmegaConf.create(
            {
                "schema_version": 1,
                "yaml_files": {"inference_config": "conf/inference.yaml"},
            }
        ),
        pack / "meta.yaml",
    )
    model = InferenceModel.from_packed(pack)
    demo = DemoSession(
        tmp_path,
        OmegaConf.create(
            {
                "model": {"call_args": {}},
                "ui": {"title": None, "description": None},
            }
        ),
        model,
        DEFAULT_UI_ASSETS.clone(),
    )
    callback = demo.create_inference_fn(demo.input_specs, demo.output_specs)
    yield config, samples, backend, model, callback


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("batch_size", [None, 2, 4])
def test_single_batch_recipe_and_demo_have_identical_text(
    shared_paths, count, batch_size
):
    """Different lengths/IDs expose order loss and final partial-batch bugs."""
    config, samples, backend, model, callback = shared_paths
    config.batch_size = batch_size
    for i in range(count):
        samples.append(
            {
                "speech": (16000, np.full(160 + 16 * i, (i + 1) / 8, dtype=np.float32)),
                "utt_id": f"utterance-{count - i}",
            }
        )
    expected = [f"word-{int(np.sum(s['speech'][1]))}!" for s in samples]
    indices = [s["utt_id"] for s in samples]
    single = [model(s, idx=s["utt_id"]) for s in samples]
    backend.calls.clear()
    batched = model.forward_batch(samples, indices=indices)
    assert len(backend.calls) == 1
    assert isinstance(backend.calls[0], list)
    assert [r["text"] for r in single] == expected
    assert batched == single
    assert [callback(s["speech"]) for s in samples] == expected
    infer(config)
    lines = config.inference_dir + "/test/text.scp"
    assert Path(lines).read_text().splitlines() == [
        f"{idx} {text}" for idx, text in zip(indices, expected)
    ]


def test_empty_public_batch_does_not_decode(shared_paths):
    """An empty batch has no backend side effect."""
    _, _, backend, model, _ = shared_paths
    assert model.forward_batch([]) == []
    assert backend.calls == []


@pytest.mark.parametrize(
    "bad",
    [
        None,
        123,
        (8000, np.zeros(160, dtype=np.float32)),
        (16000, np.zeros((2, 160), dtype=np.float32)),
        (16000, np.array([0, np.nan, 0], dtype=np.float32)),
    ],
)
def test_input_errors_are_shared_across_entry_points(shared_paths, bad):
    """Contract errors occur before decoding through each public entry point."""
    config, samples, backend, model, callback = shared_paths
    sample = {"speech": bad, "utt_id": "bad"}
    samples.append(sample)
    for call in (
        lambda: model(sample),
        lambda: model.forward_batch([{"speech": (16000, np.zeros(160))}, sample]),
        lambda: callback(bad),
        lambda: infer(config),
    ):
        with pytest.raises((TypeError, ValueError)):
            call()
        assert backend.calls == []


def test_backend_failure_then_valid_call_recovers(shared_paths):
    """A failing call must not replace a later result with stale success."""
    _, _, backend, model, callback = shared_paths
    sample = {"speech": (16000, np.ones(160, dtype=np.float32))}
    for call in (lambda: model(sample), lambda: callback(sample["speech"])):
        backend.fail = True
        with pytest.raises(RuntimeError, match="backend failed"):
            call()
        backend.fail = False
        result = call()
        assert (result["text"] if isinstance(result, dict) else result) == "word-160!"


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_worker_shard_is_not_marked_done_and_can_be_retried(
    shared_paths, workers
):
    config, samples, backend, _, _ = shared_paths
    config.parallel.n_workers = workers
    config.batch_size = 1
    samples.extend(
        [
            {"speech": (16000, np.ones(160, dtype=np.float32)), "utt_id": "valid"},
            {"speech": (16000, np.array([np.nan], dtype=np.float32)), "utt_id": "bad"},
        ]
    )
    with pytest.raises(ValueError, match="finite"):
        infer(config)
    output = Path(config.inference_dir) / "test"
    assert not (output / "text.scp").exists()
    assert any(not (shard / "done").exists() for shard in output.glob("split.*"))
    samples[1]["speech"] = (16000, np.full(160, 2, dtype=np.float32))
    infer(config)
    assert (output / "text.scp").read_text().splitlines() == [
        "valid word-160!",
        "bad word-320!",
    ]
    assert all((shard / "done").exists() for shard in output.glob("split.*"))


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_corrupt_audio_file_fails_and_session_recovers(
    shared_paths, tmp_path, exists
):
    _, _, backend, model, callback = shared_paths
    path = tmp_path / "broken.wav"
    if exists:
        path.write_bytes(b"not a WAV file")
    for call in (lambda: model({"speech": str(path)}), lambda: callback(str(path))):
        with pytest.raises((OSError, RuntimeError)):
            call()
        assert backend.calls == []
    assert (
        model({"speech": (16000, np.ones(160, dtype=np.float32))})["text"]
        == "word-160!"
    )


def test_publication_streaming_preserves_flags_and_decoder_state(shared_paths):
    from test.espnet3.systems.esp2_asr.test_asr_streaming_inference_runner import (
        StreamingBackend,
    )

    _, _, _, model, _ = shared_paths
    backend = StreamingBackend()
    model.model = backend
    assert (
        model({"speech": (16000, np.ones(1, dtype=np.float32))}, is_final=False)["text"]
        == ""
    )
    assert (
        model({"speech": (16000, np.ones(3, dtype=np.float32))}, is_final=False)["text"]
        == "4:4"
    )
    assert (
        model({"speech": (16000, np.empty(0, dtype=np.float32))}, is_final=True)["text"]
        == "4:4"
    )
    assert [final for _, final in backend.calls] == [False, False, True]
    assert backend.parts == []


def test_custom_kind_survives_packing_and_all_entry_points(
    shared_paths, tmp_path, monkeypatch
):
    from espnet3.contract import KINDS, Kind

    class Tokens(Kind):
        option_names = Kind.option_names | {"max_length"}

        def check(self, value, field, model, *, output):
            if not isinstance(value, list) or not all(
                isinstance(x, int) for x in value
            ):
                raise TypeError("tokens must be a list of integers")
            return value

        def check_configured(self, value, field, model, options, *, output):
            value = self.check(value, field, model, output=output)
            if len(value) > options["max_length"]:
                raise ValueError("too many tokens")
            return value

        def to_model_input(self, value, field):
            return tuple(value)

        def to_demo_output(self, value):
            return ":".join(map(str, value))

    class TokenResult(Tokens):
        def check_configured(self, value, field, model, options, *, output):
            if not isinstance(value, dict) or set(value) != {"values"}:
                raise TypeError("token result must contain values")
            super().check_configured(
                value["values"], field, model, options, output=output
            )
            return value

        def to_demo_output(self, value):
            return ":".join(map(str, value["values"]))

    monkeypatch.setitem(KINDS, "shared_tokens", Tokens())
    monkeypatch.setitem(KINDS, "shared_token_result", TokenResult())
    config, samples, _, _, _ = shared_paths
    config.input_key = "tokens"
    config.output_keys = ["tokens"]
    config.runner = {
        "_target_": "espnet3.systems.base.inference_runner.InferenceRunner"
    }
    declaration = {"kind": "shared_tokens", "max_length": 3}
    config.contract = {
        "inputs": {"tokens": declaration},
        "outputs": {"tokens": {"kind": "shared_token_result", "max_length": 3}},
    }
    seen = []

    def backend(tokens):
        seen.append(tokens)
        if isinstance(tokens, list):
            return [{"tokens": {"values": list(reversed(item))}} for item in tokens]
        return {"tokens": {"values": list(reversed(tokens))}}

    monkeypatch.setattr(
        InferenceProvider, "build_model", staticmethod(lambda config: backend)
    )
    pack = tmp_path / "pack"
    OmegaConf.save(config, pack / "conf" / "inference.yaml")
    model = InferenceModel.from_packed(pack)
    demo = DemoSession(
        pack,
        OmegaConf.create(
            {"model": {"call_args": {}}, "ui": {"title": None, "description": None}}
        ),
        model,
        DEFAULT_UI_ASSETS.clone(),
    )
    callback = demo.create_inference_fn(demo.input_specs, demo.output_specs)
    samples.extend([{"tokens": [1, 2], "utt_id": "a"}, {"tokens": [3], "utt_id": "b"}])
    expected = [
        {"tokens": {"values": [2, 1]}, "utt_id": "a"},
        {"tokens": {"values": [3]}, "utt_id": "b"},
    ]
    assert [model(sample) for sample in samples] == expected
    assert model.forward_batch(samples) == expected
    assert callback([1, 2]) == "2:1"
    assert seen[:3] == [(1, 2), (3,), [(1, 2), (3,)]]
    infer(config)
    assert (Path(config.inference_dir) / "test" / "tokens.scp").is_file()
    with pytest.raises(ValueError, match="too many tokens"):
        model({"tokens": [1, 2, 3, 4]})
    assert model.contract.inputs[0].options["max_length"] == 3


def test_publication_ci_checks_compare_recipe_and_native_oracle(
    shared_paths, tmp_path, monkeypatch
):
    import importlib.util
    import sys

    config, _, _, _, _ = shared_paths
    config.contract.outputs.text.pop("normalize")
    pack = tmp_path / "pack"
    OmegaConf.save(config, pack / "conf" / "inference.yaml")
    source = (
        Path(__file__).resolve().parents[3]
        / "ci"
        / "test_integration_espnet3_publication_check.py"
    )
    spec = importlib.util.spec_from_file_location("publication_ci_parity", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    sample = {"speech": (16000, np.ones(160, dtype=np.float32))}
    expected = module._run_smoke_check(InferenceModel.from_packed(pack), sample)
    assert expected == "  WORD-160!  "
    module._check_recipe(pack, sample, expected)
