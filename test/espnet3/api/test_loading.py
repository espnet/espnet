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
    ModelTagError,
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


def test_read_meta_loads_an_older_schema_with_a_warning(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        read_meta(_pack(tmp_path, schema=0))  # legacy: no version at all
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        read_meta(_pack(tmp_path / "b", schema=PACK_SCHEMA_VERSION - 1 or 0))
    if PACK_SCHEMA_VERSION > 1:
        assert "this installation writes" in caplog.text


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


def test_build_model_reads_the_model_and_never_a_provider():
    """Calls go one way: a provider calls build_model, never the reverse."""
    cfg = OmegaConf.create(
        {
            "model": {"_target_": f"{__name__}.Backend", "prefix": "model:"},
            "provider": {"_target_": f"{__name__}.StubProvider"},
        }
    )
    assert build_model(cfg).prefix == "model:"  # not StubProvider's "provider:"


def test_a_bundles_own_provider_needs_no_trust_since_loading_never_runs_it(
    tmp_path,
):
    root = _pack(
        tmp_path, extra="provider:\n  _target_: src.code.Provider\n", bundled=True
    )
    with pytest.raises(ValueError, match="trust_user_code"):
        read_bundle(root)  # it is bundled code
    assert isinstance(load_model(root), Backend)  # and load_model drops it


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


def test_read_bundle_counts_a_namespace_package_as_bundled_code(tmp_path):
    """`src/code.py` with no `src/__init__.py` is still code the target runs."""
    root = _pack(tmp_path, model_target="src.code.Local", bundled=True)
    (root / "src" / "__init__.py").unlink()
    with pytest.raises(ValueError, match="trust_user_code"):
        read_bundle(root)


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
    with pytest.raises(ModelTagError, match="not an Inference. Pass system=<name>"):
        load(_pack(tmp_path))


def test_load_passes_constructor_arguments_over_the_packed_ones(tmp_path):
    """`load(tag, beam_size=1)`, as ESPnet2's `from_pretrained` takes it."""
    root = _pack(tmp_path, model_target=f"{__name__}.Echo")
    model = load(root, prefix="mine:")
    assert model.prefix == "mine:"
    assert model(np.zeros(8000, dtype=np.float32))["text"] == "mine:1.0s@cpu"
    # nothing is kept: the next load reads the bundle as packed
    assert load(root).prefix == "cfg:"


def test_load_model_overrides_are_constructor_arguments_only(tmp_path):
    root = _pack(tmp_path)
    assert load_model(root, overrides={"prefix": "over:"}).prefix == "over:"
    assert load_model(root, overrides={}).prefix == "cfg:"
    with pytest.raises(ValueError, match=r"\['_target_'\] cannot be overridden"):
        load_model(root, overrides={"_target_": "builtins.dict"})
    with pytest.raises(TypeError, match="nope=1: the bundle's model .* takes no"):
        load_model(root, overrides={"nope": 1})


def test_a_tag_that_is_not_a_pack_is_a_model_tag_error(monkeypatch):
    """The type ESPnet2's loaders raise, so a front end reports it the same way."""
    module = types.ModuleType("espnet_model_zoo.downloader")

    class ModelDownloader:
        def download_and_unpack(self, tag):
            return {"asr_train_config": "c.yaml", "asr_model_file": "m.pth"}

    module.ModelDownloader = ModelDownloader
    monkeypatch.setitem(sys.modules, "espnet_model_zoo.downloader", module)
    with pytest.raises(ModelTagError, match="not a pack_model bundle"):
        loading.locate_pack("espnet/an_espnet2_model")
    assert issubclass(ModelTagError, RuntimeError)


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
    with pytest.raises(ModelTagError, match="system 'nosuch' has no Inference yet"):
        load(_pack(tmp_path / "b", system="nosuch"))
    _install_fake_system(monkeypatch, "blank", None)
    with pytest.raises(ImportError, match="defines no Inference"):
        load(_pack(tmp_path / "c", system="blank"))


class DataOnlyProvider(InferenceProvider):
    """A recipe provider that changes only the dataset, as most will."""

    @staticmethod
    def build_dataset(config):
        return []


def test_a_provider_that_only_builds_the_dataset_builds_the_model_once(tmp_path):
    """popcornell's case on #6806: it used to recurse until RecursionError."""
    config = OmegaConf.create(
        {
            "recipe_dir": str(tmp_path),
            "provider": {"_target_": f"{__name__}.DataOnlyProvider"},
            "model": {"_target_": f"{__name__}.Backend", "prefix": "p:"},
        }
    )
    for model in (build_model(config), DataOnlyProvider.build_model(config)):
        assert isinstance(model, Backend) and model.prefix == "p:"


def test_load_names_an_override_the_bundles_own_inference_does_not_take(tmp_path):
    """The no-system path builds as load_model does, error included."""
    root = _pack(tmp_path, model_target=f"{__name__}.Echo")
    with pytest.raises(TypeError, match="bogus=1: the bundle's model .* takes no"):
        load(root, bogus=1)


def test_a_system_bundle_with_bundled_code_is_told_to_repack(tmp_path):
    """trust_user_code cannot help it: a system's Inference runs no bundle code."""
    root = _pack(
        tmp_path, model_target="src.code.Local", system="esp2_asr", bundled=True
    )
    with pytest.raises(ValueError, match="names system 'esp2_asr'.*Re-pack"):
        read_bundle(root)
