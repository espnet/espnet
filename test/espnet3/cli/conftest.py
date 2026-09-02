"""Shared fixtures for clone CLI tests."""

from __future__ import annotations

import pytest

import espnet3.cli.clone.resolver as resolver_module


@pytest.fixture
def fake_egs3(tmp_path, monkeypatch):
    """Create a small egs3 layout and make the resolver use it."""
    for path in (
        "mini_an4/asr",
        "librispeech/asr",
        "TEMPLATE/asr",
        ".hidden/asr",
        "__pycache__",
    ):
        (tmp_path / path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(resolver_module, "_get_egs3_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def empty_egs3(tmp_path, monkeypatch):
    """Make the resolver use an empty egs3 layout."""
    monkeypatch.setattr(resolver_module, "_get_egs3_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def fake_recipe(tmp_path):
    """Create a recipe with every included item and excluded extras."""
    recipe = tmp_path / "recipe_src"
    recipe.mkdir()
    for directory in ("conf", "src", "dataset"):
        subdir = recipe / directory
        subdir.mkdir()
        (subdir / "file.py").write_text("# content")
        pycache = subdir / "__pycache__"
        pycache.mkdir()
        (pycache / "file.cpython-311.pyc").write_bytes(b"")
    (recipe / "run.py").write_text("# run")
    (recipe / "readme.md").write_text("# readme")
    (recipe / "path.sh").write_text("#!/bin/bash")
    (recipe / "demo").mkdir()
    (recipe / "demo" / "app.py").write_text("# demo")
    (recipe / "__init__.py").write_text("")
    (recipe / "__pycache__").mkdir()
    return recipe
