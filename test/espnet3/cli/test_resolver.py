"""Tests for :mod:`espnet3.cli.clone.resolver`."""

from __future__ import annotations

import pytest

from espnet3.cli.clone.resolver import _list_available, list_recipes, resolve_recipe


def test_resolve_recipe_returns_correct_path(fake_egs3):
    assert resolve_recipe("mini_an4/asr") == fake_egs3 / "mini_an4" / "asr"


@pytest.mark.parametrize("recipe", ("mini_an4", "a/b/c", "", "/", "mini_an4/"))
def test_resolve_recipe_rejects_invalid_format(recipe):
    with pytest.raises(ValueError, match="<dataset>/<task>"):
        resolve_recipe(recipe)


def test_resolve_recipe_rejects_path_outside_egs3(fake_egs3):
    (fake_egs3.parent / "espnet3").mkdir()
    with pytest.raises(ValueError, match="inside the egs3 directory"):
        resolve_recipe("../espnet3")


def test_resolve_recipe_rejects_egs3_root(fake_egs3):
    with pytest.raises(ValueError, match="subdirectory of egs3"):
        resolve_recipe("./.")


@pytest.mark.parametrize("recipe", ("unknown/asr", "mini_an4/tts", "nosuchdataset/asr"))
def test_resolve_recipe_rejects_missing_recipe(fake_egs3, recipe):
    with pytest.raises(FileNotFoundError, match=recipe):
        resolve_recipe(recipe)


def test_resolve_recipe_error_lists_available_recipes(fake_egs3):
    with pytest.raises(FileNotFoundError) as error:
        resolve_recipe("unknown/asr")
    assert "librispeech/asr" in str(error.value)
    assert "mini_an4/asr" in str(error.value)


def test_resolve_recipe_accepts_surrounding_slashes(fake_egs3):
    assert resolve_recipe("/mini_an4/asr/") == fake_egs3 / "mini_an4" / "asr"


def test_list_recipes_returns_sorted_visible_recipes(fake_egs3):
    assert list_recipes() == ["librispeech/asr", "mini_an4/asr"]


@pytest.mark.parametrize(
    "path",
    ("not_a_dir.txt", "empty_dataset", "mini_an4/_private", "mini_an4/README.md"),
)
def test_list_available_ignores_non_recipe_entries(tmp_path, path):
    (tmp_path / "mini_an4" / "asr").mkdir(parents=True)
    target = tmp_path / path
    if target.suffix:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("file")
    else:
        target.mkdir(parents=True, exist_ok=True)
    assert path not in _list_available(tmp_path)
    assert "mini_an4/asr" in _list_available(tmp_path)


def test_list_available_returns_empty_for_empty_egs3(tmp_path):
    assert _list_available(tmp_path) == []
