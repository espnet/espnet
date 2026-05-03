from __future__ import annotations

from types import SimpleNamespace

from espnet3.systems.base.publication import get_pack_model_artifacts


def test_get_pack_model_artifacts_returns_empty_artifact_dict():
    system = SimpleNamespace()

    artifacts = get_pack_model_artifacts(system)

    assert artifacts == {"files": {}, "yaml_files": {}, "copy_paths": []}


def test_get_pack_model_artifacts_has_required_keys():
    system = SimpleNamespace()

    artifacts = get_pack_model_artifacts(system)

    assert "files" in artifacts
    assert "yaml_files" in artifacts
    assert "copy_paths" in artifacts


def test_get_pack_model_artifacts_ignores_system_attributes():
    system = SimpleNamespace(training_config=object(), some_attr="value")

    artifacts = get_pack_model_artifacts(system)

    assert artifacts["files"] == {}
    assert artifacts["yaml_files"] == {}
    assert artifacts["copy_paths"] == []
