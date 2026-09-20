"""Tests for egs3/ami/s2t/conf/publication.yaml."""

import importlib.util
from pathlib import Path

import ami_sot_paths


def _config():
    """Load the config the way run.py does."""
    from espnet3.utils.config_utils import load_and_merge_config

    return load_and_merge_config(
        ami_sot_paths.RECIPE / "conf" / "publication.yaml",
        config_name="publication.yaml",
        default_package="egs3.TEMPLATE.asr",
        resolve=True,
    )


def _builder():
    spec = importlib.util.spec_from_file_location(
        "ami_s2t_builder_for_pub", ami_sot_paths.RECIPE / "dataset" / "builder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_packed_bundle_carries_the_recipe_local_modules():
    """The packed bundle carries the recipe local modules.

    A packed model imports src.preprocessor and src.inference by name, and dataset/
    supplies the builder the publication smoke check runs.
    """
    include = [str(p) for p in _config()["pack_model"]["include"]]
    assert any(p.rstrip("/").endswith("/src") or p == "src" for p in include), include
    assert any(
        p.rstrip("/").endswith("/dataset") or p == "dataset" for p in include
    ), include


def test_the_packed_bundle_carries_the_token_list(monkeypatch):
    """The packed bundle carries the token list.

    Without it the bundle cannot be decoded anywhere else: the checkpoint's config.yaml
    names the token list by absolute path into the corpus root.

    The variable has to be set for this to mean anything: with it unset both expressions
    collapse to the recipe directory and the test would pass against the very mismatch
    it exists to catch.
    """
    monkeypatch.setenv("AMI_SOT_DATA_ROOT", "/nonexistent-root-for-this-test")
    written = (
        Path(_builder()._CONFIG["data_root"]) / _builder()._CONFIG["token_list"]
    ).resolve()
    packed = {Path(str(p)).resolve() for p in _config()["pack_model"]["include"]}
    assert written in packed, sorted(str(p) for p in packed)


def test_the_checkpoint_config_is_registered_for_path_rewriting():
    """pack_model rewrites paths only inside yaml_files entries.

    The trained checkpoint's config.yaml holds an absolute token_list pointing at this
    machine's corpus root. Bulk-copied it would travel unchanged and the bundle would be
    undecodable elsewhere.
    """
    yaml_files = _config()["pack_model"].get("yaml_files") or {}
    values = [str(v) for v in yaml_files.values()]
    assert any(v.endswith("config.yaml") for v in values), yaml_files


def test_checkpoints_are_kept_but_lightning_snapshots_are_not():
    exclude = [str(p) for p in _config()["pack_model"]["exclude"]]
    assert any("ckpt" in p for p in exclude), exclude
    assert not any(p.endswith(".pth") for p in exclude), exclude
