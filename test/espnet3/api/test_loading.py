"""Reading a bundle and building its model: the one loader every caller uses."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

import espnet3.api.inference.loading as loading
from espnet3.api.inference import (
    Field,
    InferenceAPI,
    build_model,
    load,
    load_model,
    read_bundle,
    read_meta,
)
from espnet3.publication.schema import PACK_SCHEMA_VERSION
from espnet3.systems.base.inference_provider import InferenceProvider


class Echo(InferenceAPI):
    """An Inference a bundle may name as its model."""

    inputs = (Field("speech", "audio"),)
    outputs = (Field("text", "text"),)

    def __init__(self, device="cpu", prefix=""):
        self.device = device
        self.prefix = prefix

    @classmethod
    def from_pretrained(cls, tag_or_dir, *, device="cpu", **kwargs):
        return cls(device=device)

    sample_rate = 8000

    def run(self, speech):
        return {"text": f"{self.prefix}{speech.duration:.1f}s@{self.device}"}


class Backend:
    """A bare backend, as an ESPnet2 bundle names."""

    def __init__(self, device="cpu", prefix=""):
        self.device = device
        self.prefix = prefix


class StubProvider:
    """A bundle's own way of building, as the CI demo packs use."""

    @staticmethod
    def build_model(config):
        return Backend(prefix="provider:")


def _pack(
    tmp_path: Path,
    *,
    model_target: str = f"{__name__}.Backend",
    system: str | None = None,
    schema: int | None = PACK_SCHEMA_VERSION,
    extra: str = "",
    bundled: bool = False,
) -> Path:
    root = tmp_path / "pack"
    (root / "conf").mkdir(parents=True)
    meta = {"yaml_files": {"inference_config": "conf/inference.yaml"}}
    if system:
        meta["system"] = system
    if schema is not None:
        meta["schema_version"] = schema
    (root / "meta.yaml").write_text(yaml.safe_dump(meta))
    (root / "conf" / "inference.yaml").write_text(
        f"recipe_dir: .\nmodel:\n  _target_: {model_target}\n  prefix: 'cfg:'\n{extra}"
    )
    if bundled:
        (root / "src").mkdir()
        (root / "src" / "__init__.py").write_text("")
        (root / "src" / "code.py").write_text(
            "from espnet3.api.inference import Field, InferenceAPI\n"
            "class Local(InferenceAPI):\n"
            "    inputs = (Field('speech', 'audio'),)\n"
            "    outputs = (Field('text', 'text'),)\n"
            "    def __init__(self, device='cpu', prefix=''):\n"
            "        self.device = device\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, t, *, device='cpu', **k):\n"
            "        return cls()\n"
            "    sample_rate = 8000\n"
            "    def run(self, speech):\n"
            "        return {'text': 'local'}\n"
        )
    return root


# --- read_meta / read_bundle -----------------------------------------------


def test_read_meta_checks_the_directory_and_the_schema(tmp_path, caplog):
    with pytest.raises(FileNotFoundError, match="pack_dir must point"):
        read_meta(tmp_path / "nowhere")
    (tmp_path / "file").write_text("x")
    with pytest.raises(FileNotFoundError, match="pack_dir must point"):
        read_meta(tmp_path / "file")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="must contain meta.yaml"):
        read_meta(tmp_path / "empty")
    with pytest.raises(ValueError, match="newer pack_model"):
        read_meta(_pack(tmp_path, schema=PACK_SCHEMA_VERSION + 1))


def test_read_meta_warns_about_a_legacy_bundle_only(tmp_path, caplog):
    import logging

    root = _pack(tmp_path, schema=None)
    with caplog.at_level(logging.WARNING):
        read_meta(root)
    assert "no schema_version" in caplog.text
    caplog.clear()
    root2 = _pack(tmp_path / "b")
    with caplog.at_level(logging.WARNING):
        read_meta(root2)
    assert "no schema_version" not in caplog.text


def test_read_bundle_needs_the_inference_config(tmp_path):
    root = _pack(tmp_path)
    (root / "meta.yaml").write_text(yaml.safe_dump({"schema_version": 1}))
    with pytest.raises(FileNotFoundError, match="yaml_files.inference_config"):
        read_bundle(root)
    root2 = _pack(tmp_path / "b")
    (root2 / "conf" / "inference.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="listed in meta.yaml not found"):
        read_bundle(root2)


def test_read_bundle_binds_recipe_dir_and_guards_bundled_code(tmp_path):
    root = _pack(tmp_path)
    config, bundle_root = read_bundle(root)
    assert bundle_root == root.resolve() and config.recipe_dir == str(root.resolve())
    bundled = _pack(tmp_path / "b", model_target="src.code.Local", bundled=True)
    with pytest.raises(ValueError, match="trust_user_code=True"):
        read_bundle(bundled)
    config, _ = read_bundle(bundled, trust_user_code=True)
    assert str(bundled.resolve()) in sys.path
    assert config.model._target_ == "src.code.Local"


# --- build_model -----------------------------------------------------------


def test_build_model_instantiates_on_the_device():
    cfg = OmegaConf.create(
        {"model": {"_target_": f"{__name__}.Backend", "prefix": "p"}}
    )
    model = build_model(cfg, device="cuda:1")
    assert (
        isinstance(model, Backend) and model.device == "cuda:1" and model.prefix == "p"
    )
    assert build_model(cfg).device == "cpu"  # the default when nothing says
    cfg.device = "mps"
    assert build_model(cfg).device == "mps"  # the config's own device
    with pytest.raises(ValueError, match="no `model`"):
        build_model(OmegaConf.create({}))


def test_build_model_honours_a_bundles_own_provider():
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "builtins.object"},
            "provider": {"_target_": f"{__name__}.StubProvider"},
        }
    )
    assert build_model(cfg).prefix == "provider:"
    cfg.provider._target_ = loading._DEFAULT_PROVIDER
    cfg.model = {"_target_": f"{__name__}.Backend"}
    assert isinstance(build_model(cfg), Backend)  # the default provider: built here


def test_the_infer_stage_provider_builds_through_the_same_function(monkeypatch):
    seen = {}

    def fake(config, *, device=None):
        seen["device"] = device
        return "built"

    import espnet3.systems.base.inference_provider as provider_mod

    monkeypatch.setattr(provider_mod, "build_model", fake)
    cfg = OmegaConf.create({"device": "cpu", "model": {"_target_": "x.Y"}})
    assert InferenceProvider.build_model(cfg) == "built"
    assert seen["device"] == "cpu"


def test_build_model_absolutises_relative_paths_it_built_from(tmp_path):
    (tmp_path / "tokens.txt").write_text("a\n")

    class Holder:
        def __init__(self, device="cpu", path="tokens.txt"):
            self.path = path

    cfg = OmegaConf.create(
        {
            "recipe_dir": str(tmp_path),
            "model": {"_target_": f"{__name__}.Holder", "path": "tokens.txt"},
        }
    )
    globals()["Holder"] = Holder
    model = build_model(cfg)
    assert Path(model.path).is_absolute() and Path(model.path).exists()


# --- load_model ------------------------------------------------------------


def test_load_model_drops_output_fn_and_runner_before_the_trust_check(tmp_path):
    root = _pack(
        tmp_path,
        extra=(
            "output_fn: src.code.build_output\n"
            "runner:\n  _target_: src.code.Runner\n"
        ),
        bundled=True,
    )
    with pytest.raises(ValueError, match="trust_user_code"):
        read_bundle(root)
    model = load_model(root, device="cuda:0")
    assert isinstance(model, Backend) and model.device == "cuda:0"


def test_load_model_refuses_a_model_that_is_bundled_code(tmp_path):
    root = _pack(tmp_path, model_target="src.code.Local", bundled=True)
    with pytest.raises(ValueError, match="trust_user_code"):
        load_model(root)
    assert load_model(root, trust_user_code=True).run(None)["text"] == "local"


# --- load ------------------------------------------------------------------


def _install_fake_system(monkeypatch, name, cls):
    module = types.ModuleType(f"espnet3.systems.{name}.inference")
    if cls is not None:
        module.Inference = cls
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_load_goes_through_the_system_named_in_meta(tmp_path, monkeypatch):
    _install_fake_system(monkeypatch, "echo", Echo)
    model = load(_pack(tmp_path, system="echo"), device="cuda:0")
    assert isinstance(model, Echo) and model.device == "cuda:0"


def test_load_returns_the_bundles_own_inference_when_no_system_is_named(tmp_path):
    root = _pack(tmp_path, model_target=f"{__name__}.Echo")
    model = load(root)
    assert isinstance(model, Echo) and model.prefix == "cfg:"
    assert model(np.zeros(8000, dtype=np.float32))["text"] == "cfg:1.0s@cpu"


def test_load_says_when_neither_a_system_nor_an_inference_is_there(tmp_path):
    with pytest.raises(ValueError, match="not an Inference. Pass system=<name>"):
        load(_pack(tmp_path))


def test_load_trusts_a_bundles_own_inference_only_when_told(tmp_path):
    root = _pack(tmp_path, model_target="src.code.Local", bundled=True)
    with pytest.raises(ValueError, match="trust_user_code"):
        load(root)
    assert load(root, trust_user_code=True).run(None)["text"] == "local"


def test_load_follows_an_alias_and_names_a_missing_system(tmp_path, monkeypatch):
    _install_fake_system(monkeypatch, "esp2_echo", Echo)
    monkeypatch.setitem(loading.SYSTEM_ALIASES, "echo", "esp2_echo")
    assert isinstance(load(_pack(tmp_path, system="echo")), Echo)
    assert loading.SYSTEM_ALIASES["asr"] == "esp2_asr"
    with pytest.raises(ImportError, match="system 'nosuch' has no Inference yet"):
        load(_pack(tmp_path / "b", system="nosuch"))
    _install_fake_system(monkeypatch, "blank", None)
    with pytest.raises(ImportError, match="defines no Inference"):
        load(_pack(tmp_path / "c", system="blank"))
