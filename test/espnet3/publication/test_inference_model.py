from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from espnet3.publication import InferenceModel
from espnet3.systems.base.inference_provider import InferenceProvider
from espnet3.systems.base.inference_runner import InferenceRunner


class EchoModel:
    def __init__(self, device="cpu", prefix=""):
        self.device = device
        self.prefix = prefix

    def __call__(self, speech):
        return f"{self.prefix}{speech}@{self.device}"


class DualInputModel:
    def __call__(self, speech, text):
        return f"{speech}+{text}"


class BatchEchoModel:
    def __init__(self):
        self.calls = []

    def __call__(self, speech):
        self.calls.append(speech)
        if isinstance(speech, list):
            return [f"batched:{item}" for item in speech]
        return f"single:{speech}"


def _make_pack_dir(
    tmp_path: Path,
    *,
    model_target: str = "builtins.object",
    input_key: str | list = "speech",
    output_fn: str | None = None,
    with_src_module: bool = False,
    provider_target: str | None = None,
    runner_target: str | None = None,
) -> Path:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    conf_dir = bundle_root / "conf"
    conf_dir.mkdir()

    lines = ["recipe_dir: ."]
    if isinstance(input_key, list):
        lines.append("input_key:")
        for k in input_key:
            lines.append(f"  - {k}")
    else:
        lines.append(f"input_key: {input_key}")
    lines += [
        "model:",
        f"  _target_: {model_target}",
        "  prefix: 'cfg:'",
    ]
    if provider_target is not None:
        lines += [
            "provider:",
            f"  _target_: {provider_target}",
        ]
    if runner_target is not None:
        lines += [
            "runner:",
            f"  _target_: {runner_target}",
        ]
    if output_fn is not None:
        lines.append(f"output_fn: {output_fn}")
    (conf_dir / "inference.yaml").write_text("\n".join(lines), encoding="utf-8")

    with (bundle_root / "meta.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {"files": {}, "yaml_files": {"inference_config": "conf/inference.yaml"}},
            stream,
        )
    if with_src_module:
        src_dir = bundle_root / "src"
        src_dir.mkdir()
        (src_dir / "__init__.py").write_text("", encoding="utf-8")
        (src_dir / "custom_code.py").write_text(
            "\n".join(
                [
                    "from espnet3.systems.base.inference_provider"
                    " import InferenceProvider",
                    "from espnet3.systems.base.inference_runner import InferenceRunner",
                    "",
                    "class CustomModel:",
                    "    def __init__(self, prefix='', device='cpu'):",
                    "        self.prefix = prefix",
                    "        self.device = device",
                    "",
                    "    def __call__(self, speech):",
                    "        return f'{self.prefix}{speech}'",
                    "",
                    "class CustomProvider(InferenceProvider):",
                    "    @staticmethod",
                    "    def build_model(config):",
                    "        return CustomModel(prefix='provider:')",
                    "",
                    "class CustomRunner(InferenceRunner):",
                    "    @staticmethod",
                    "    def forward(idx, dataset=None, model=None, **kwargs):",
                    "        out = InferenceRunner.forward(",
                    "            idx, dataset=dataset, model=model, **kwargs",
                    "        )",
                    "        if isinstance(out, list):",
                    "            return [f'runner:{item}' for item in out]",
                    "        return f'runner:{out}'",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return bundle_root


@pytest.fixture
def mock_build_model(monkeypatch):
    """Patch InferenceProvider.build_model to return a simple EchoModel."""

    def _fake_build(config):
        prefix = getattr(getattr(config, "model", None), "prefix", "")
        return EchoModel(prefix=prefix)

    monkeypatch.setattr(InferenceProvider, "build_model", staticmethod(_fake_build))


@pytest.fixture
def mock_build_dual_model(monkeypatch):
    """Patch InferenceProvider.build_model to return a DualInputModel."""
    monkeypatch.setattr(
        InferenceProvider,
        "build_model",
        staticmethod(lambda config: DualInputModel()),
    )


@pytest.fixture
def mock_build_batch_model(monkeypatch):
    """Patch InferenceProvider.build_model to return a batch-friendly model."""
    model = BatchEchoModel()
    monkeypatch.setattr(
        InferenceProvider,
        "build_model",
        staticmethod(lambda config: model),
    )
    return model


# ---------------------------------------------------------------------------
# schema_version helpers
# ---------------------------------------------------------------------------


def _set_schema_version(bundle_root: Path, version: int | None) -> None:
    meta_path = bundle_root / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    if version is None:
        meta.pop("schema_version", None)
    else:
        meta["schema_version"] = version
    meta_path.write_text(yaml.dump(meta), encoding="utf-8")


# ---------------------------------------------------------------------------
# from_packed – error cases
# ---------------------------------------------------------------------------


def test_from_packed_raises_if_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="pack_dir must point"):
        InferenceModel.from_packed(tmp_path / "nonexistent")


def test_from_packed_raises_if_pack_dir_is_file(tmp_path):
    fake_file = tmp_path / "not_a_dir"
    fake_file.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="pack_dir must point"):
        InferenceModel.from_packed(fake_file)


def test_from_packed_raises_if_meta_missing(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    with pytest.raises(FileNotFoundError, match="meta.yaml"):
        InferenceModel.from_packed(bundle_root)


def test_from_packed_raises_if_inference_config_not_in_meta(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    with (bundle_root / "meta.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"files": {}, "yaml_files": {}}, stream)
    with pytest.raises(FileNotFoundError, match="yaml_files.inference_config"):
        InferenceModel.from_packed(bundle_root)


def test_from_packed_raises_if_inference_config_file_missing(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    with (bundle_root / "meta.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {"files": {}, "yaml_files": {"inference_config": "conf/inference.yaml"}},
            stream,
        )
    with pytest.raises(FileNotFoundError):
        InferenceModel.from_packed(bundle_root)


def test_from_packed_raises_without_trust_for_bundled_code(tmp_path):
    bundle_root = _make_pack_dir(
        tmp_path,
        model_target="src.custom_code.CustomModel",
        with_src_module=True,
    )
    with pytest.raises(ValueError, match="references bundled user code"):
        InferenceModel.from_packed(bundle_root, trust_user_code=False)


def test_from_packed_raises_for_future_schema_version(tmp_path, mock_build_model):
    bundle_root = _make_pack_dir(tmp_path)
    _set_schema_version(bundle_root, 999)
    with pytest.raises(ValueError, match="schema_version=999"):
        InferenceModel.from_packed(bundle_root)


def test_from_packed_warns_for_legacy_bundle_without_schema_version(
    tmp_path, mock_build_model
):
    bundle_root = _make_pack_dir(tmp_path)
    _set_schema_version(bundle_root, None)

    with patch("espnet3.publication.inference_model.logger") as mock_logger:
        InferenceModel.from_packed(bundle_root)

    mock_logger.warning.assert_called()
    args, _ = mock_logger.warning.call_args
    assert "legacy" in args[0].lower() or "schema_version" in args[0].lower()


def test_from_packed_no_warning_for_current_schema_version(tmp_path, mock_build_model):
    bundle_root = _make_pack_dir(tmp_path)
    _set_schema_version(bundle_root, 1)

    with patch("espnet3.publication.inference_model.logger") as mock_logger:
        InferenceModel.from_packed(bundle_root)

    schema_warnings = [
        call
        for call in mock_logger.warning.call_args_list
        if "schema" in str(call).lower() or "legacy" in str(call).lower()
    ]
    assert schema_warnings == []


# ---------------------------------------------------------------------------
# from_packed – success cases
# ---------------------------------------------------------------------------


def test_from_packed_loads_model_without_bundled_code(tmp_path, mock_build_model):
    bundle_root = _make_pack_dir(tmp_path)
    session = InferenceModel.from_packed(bundle_root, trust_user_code=False)
    assert session("hi") == "cfg:hi@cpu"


def test_from_packed_with_trust_user_code(tmp_path, monkeypatch):
    bundle_root = _make_pack_dir(
        tmp_path,
        model_target="src.custom_code.CustomModel",
        with_src_module=True,
    )
    monkeypatch.setattr(sys, "path", list(sys.path))

    session = InferenceModel.from_packed(bundle_root, trust_user_code=True)

    assert session({"speech": "hello"}) == "cfg:hello"


def test_from_packed_uses_configured_provider_and_runner_targets(tmp_path, monkeypatch):
    bundle_root = _make_pack_dir(
        tmp_path,
        model_target="src.custom_code.CustomModel",
        with_src_module=True,
        provider_target="src.custom_code.CustomProvider",
        runner_target="src.custom_code.CustomRunner",
    )
    monkeypatch.setattr(sys, "path", list(sys.path))

    session = InferenceModel.from_packed(bundle_root, trust_user_code=True)

    assert session("hello") == "runner:provider:hello"


def test_from_packed_does_not_add_bundle_to_path_without_bundled_code(
    tmp_path, mock_build_model
):
    bundle_root = _make_pack_dir(tmp_path)
    original_path = list(sys.path)

    InferenceModel.from_packed(bundle_root, trust_user_code=True)

    assert sys.path == original_path


# ---------------------------------------------------------------------------
# from_pretrained
# ---------------------------------------------------------------------------


def test_from_pretrained_raises_if_no_inference_config(monkeypatch):
    class FakeDownloader:
        def download_and_unpack(self, _tag):
            return {"model_file": "/some/file"}

    monkeypatch.setattr(
        "espnet3.publication.inference_model.ModelDownloader",
        FakeDownloader,
    )

    with pytest.raises(RuntimeError, match="inference_config"):
        InferenceModel.from_pretrained("dummy/tag")


def test_from_pretrained_loads_via_from_packed(tmp_path, monkeypatch, mock_build_model):
    bundle_root = _make_pack_dir(tmp_path)
    inference_config_path = bundle_root / "conf" / "inference.yaml"

    class FakeDownloader:
        def download_and_unpack(self, _tag):
            return {"inference_config": str(inference_config_path)}

    monkeypatch.setattr(
        "espnet3.publication.inference_model.ModelDownloader",
        FakeDownloader,
    )
    monkeypatch.setattr(sys, "path", list(sys.path))

    session = InferenceModel.from_pretrained("dummy/tag")

    assert session("ok") == "cfg:ok@cpu"


# ---------------------------------------------------------------------------
# primary_input_key
# ---------------------------------------------------------------------------


def test_primary_input_key_returns_string_key(tmp_path, mock_build_model):
    bundle_root = _make_pack_dir(tmp_path, input_key="speech")
    session = InferenceModel.from_packed(bundle_root)
    assert session.primary_input_key == "speech"


def test_primary_input_key_returns_key_from_single_element_list(
    tmp_path, mock_build_model
):
    bundle_root = _make_pack_dir(tmp_path, input_key=["speech"])
    session = InferenceModel.from_packed(bundle_root)
    assert session.primary_input_key == "speech"


def test_primary_input_key_raises_for_multiple_keys(tmp_path, mock_build_dual_model):
    bundle_root = _make_pack_dir(tmp_path, input_key=["speech", "text"])
    session = InferenceModel.from_packed(bundle_root)
    with pytest.raises(RuntimeError, match="exactly one configured input_key"):
        _ = session.primary_input_key


# ---------------------------------------------------------------------------
# forward – input handling
# ---------------------------------------------------------------------------


def test_forward_with_scalar_input(tmp_path, mock_build_model):
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))
    assert session("hello") == "cfg:hello@cpu"


def test_forward_with_mapping_input(tmp_path, mock_build_model):
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))
    assert session({"speech": "world"}) == "cfg:world@cpu"


def test_forward_raises_for_missing_key_in_mapping(tmp_path, mock_build_model):
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))
    with pytest.raises(KeyError, match="speech"):
        session({"text": "no-speech-key"})


def test_forward_with_scalar_raises_for_multiple_input_keys(
    tmp_path, mock_build_dual_model
):
    bundle_root = _make_pack_dir(tmp_path, input_key=["speech", "text"])
    session = InferenceModel.from_packed(bundle_root)
    with pytest.raises(RuntimeError, match="exactly one configured input_key"):
        session("scalar-value")


def test_forward_with_multiple_input_keys_mapping(tmp_path, mock_build_dual_model):
    bundle_root = _make_pack_dir(tmp_path, input_key=["speech", "text"])
    session = InferenceModel.from_packed(bundle_root)
    result = session({"speech": "hi", "text": "bye"})
    assert result == "hi+bye"


def test_forward_raises_for_missing_key_in_multi_key_mapping(
    tmp_path, mock_build_dual_model
):
    bundle_root = _make_pack_dir(tmp_path, input_key=["speech", "text"])
    session = InferenceModel.from_packed(bundle_root)
    with pytest.raises(KeyError, match="text"):
        session({"speech": "hi"})


# ---------------------------------------------------------------------------
# forward – output_fn
# ---------------------------------------------------------------------------


def test_forward_applies_output_fn(tmp_path, mock_build_model, monkeypatch):
    bundle_root = _make_pack_dir(tmp_path, output_fn="dummy.module.fn")

    def fake_load(path):
        def fn(*, data, model_output, idx):
            return {"hyp": model_output, "idx": idx}

        return fn

    monkeypatch.setattr(
        "espnet3.publication.inference_model._load_output_fn", fake_load
    )

    session = InferenceModel.from_packed(bundle_root)
    result = session("hello", idx=42)

    assert result["hyp"] == "cfg:hello@cpu"
    assert result["idx"] == 42


def test_forward_without_output_fn_returns_raw_model_output(tmp_path, mock_build_model):
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path, output_fn=None))
    assert session("abc") == "cfg:abc@cpu"


def test_forward_passes_idx_to_output_fn(tmp_path, mock_build_model, monkeypatch):
    bundle_root = _make_pack_dir(tmp_path, output_fn="dummy.module.fn")
    captured = []

    def fake_load(path):
        def fn(*, data, model_output, idx):
            captured.append(idx)
            return model_output

        return fn

    monkeypatch.setattr(
        "espnet3.publication.inference_model._load_output_fn", fake_load
    )

    session = InferenceModel.from_packed(bundle_root)
    session("x", idx="utt-99")

    assert captured == ["utt-99"]


# ---------------------------------------------------------------------------
# forward_batch
# ---------------------------------------------------------------------------


def test_forward_batch_returns_one_result_per_sample(tmp_path, mock_build_model):
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))
    results = session.forward_batch(["a", "b", "c"])
    assert results == ["cfg:a@cpu", "cfg:b@cpu", "cfg:c@cpu"]


def test_forward_batch_empty_input_returns_empty_list(tmp_path, mock_build_model):
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))
    assert session.forward_batch([]) == []


def test_forward_batch_rejects_length_mismatch(tmp_path, mock_build_model):
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))
    with pytest.raises(ValueError, match="same length as samples"):
        session.forward_batch(["a"], indices=["x", "y"])


def test_forward_batch_uses_custom_indices(tmp_path, mock_build_model, monkeypatch):
    bundle_root = _make_pack_dir(tmp_path, output_fn="dummy.module.fn")
    captured = []

    def fake_load(path):
        def fn(*, data, model_output, idx):
            captured.append(idx)
            return model_output

        return fn

    monkeypatch.setattr(
        "espnet3.publication.inference_model._load_output_fn", fake_load
    )

    session = InferenceModel.from_packed(bundle_root)
    session.forward_batch(["a", "b"], indices=[10, 20])

    assert captured == [[10, 20], 10, 20]


def test_forward_batch_defaults_indices_to_range(
    tmp_path, mock_build_model, monkeypatch
):
    bundle_root = _make_pack_dir(tmp_path, output_fn="dummy.module.fn")
    captured = []

    def fake_load(path):
        def fn(*, data, model_output, idx):
            captured.append(idx)
            return model_output

        return fn

    monkeypatch.setattr(
        "espnet3.publication.inference_model._load_output_fn", fake_load
    )

    session = InferenceModel.from_packed(bundle_root)
    session.forward_batch(["a", "b", "c"])

    assert captured == [[0, 1, 2], 0, 1, 2]


def test_forward_batch_uses_runner_batched_path_when_supported(
    tmp_path, mock_build_batch_model
):
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))

    results = session.forward_batch(["a", "b", "c"])

    assert results == ["batched:a", "batched:b", "batched:c"]
    assert mock_build_batch_model.calls == [["a", "b", "c"]]


def _make_raising_runner(exc_factory):
    """Return a patched InferenceRunner.forward that raises on batched calls."""
    original = InferenceRunner.forward

    def patched(idx, dataset=None, model=None, **kwargs):
        if isinstance(idx, list):
            raise exc_factory()
        return original(idx, dataset=dataset, model=model, **kwargs)

    return staticmethod(patched)


def test_forward_batch_falls_back_on_type_error(
    tmp_path, mock_build_model, monkeypatch
):
    monkeypatch.setattr(
        InferenceRunner, "forward", _make_raising_runner(lambda: TypeError("no batch"))
    )
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))

    with patch("espnet3.publication.inference_model.logger") as mock_logger:
        results = session.forward_batch(["a", "b"])

    mock_logger.debug.assert_called_once()
    _, kwargs = mock_logger.debug.call_args
    assert kwargs.get("exc_info") is True
    assert results == ["cfg:a@cpu", "cfg:b@cpu"]


def test_forward_batch_falls_back_on_not_implemented_error(
    tmp_path, mock_build_model, monkeypatch
):
    monkeypatch.setattr(
        InferenceRunner,
        "forward",
        _make_raising_runner(lambda: NotImplementedError("no batch")),
    )
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))

    with patch("espnet3.publication.inference_model.logger") as mock_logger:
        results = session.forward_batch(["a", "b"])

    mock_logger.debug.assert_called_once()
    _, kwargs = mock_logger.debug.call_args
    assert kwargs.get("exc_info") is True
    assert results == ["cfg:a@cpu", "cfg:b@cpu"]


def test_forward_batch_falls_back_on_runtime_error(
    tmp_path, mock_build_model, monkeypatch
):
    monkeypatch.setattr(
        InferenceRunner,
        "forward",
        _make_raising_runner(lambda: RuntimeError("CUDA OOM")),
    )
    session = InferenceModel.from_packed(_make_pack_dir(tmp_path))

    with patch("espnet3.publication.inference_model.logger") as mock_logger:
        results = session.forward_batch(["a", "b"])

    mock_logger.warning.assert_called_once()
    args, _ = mock_logger.warning.call_args
    assert "CUDA OOM" in str(args[1])
    assert results == ["cfg:a@cpu", "cfg:b@cpu"]
