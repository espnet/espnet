"""Tests for espnet3.cli.clone (resolver and command)."""

from __future__ import annotations

import argparse

import pytest

import espnet3.cli.clone.resolver as resolver_module
from espnet3.cli.clone.command import (
    _INCLUDE,
    _copy_recipe,
    _inject_corpus_system,
    add_arguments,
    run,
)
from espnet3.cli.clone.resolver import _list_available, list_recipes, resolve_recipe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_egs3(tmp_path, monkeypatch):
    """Fake egs3/ layout with two recipes plus noise entries.

    Structure::

        <tmp>/
        ├── mini_an4/asr/
        ├── librispeech/asr/
        ├── TEMPLATE/asr/        <- must be skipped
        ├── .hidden/asr/         <- must be skipped
        └── __pycache__/         <- must be skipped
    """
    for path in [
        "mini_an4/asr",
        "librispeech/asr",
        "TEMPLATE/asr",
        ".hidden/asr",
        "__pycache__",
    ]:
        (tmp_path / path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(resolver_module, "_get_egs3_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def fake_recipe(tmp_path):
    """Minimal recipe directory containing all _INCLUDE entries plus extras.

    Extras (demo/, __init__.py, __pycache__/) must NOT appear in the clone.
    """
    recipe = tmp_path / "recipe_src"
    recipe.mkdir()

    # directories
    for d in ("conf", "src", "dataset"):
        subdir = recipe / d
        subdir.mkdir()
        (subdir / "file.py").write_text("# content")
        pycache = subdir / "__pycache__"
        pycache.mkdir()
        (pycache / "file.cpython-311.pyc").write_bytes(b"")

    # files
    (recipe / "run.py").write_text("# run")
    (recipe / "readme.md").write_text("# readme")
    (recipe / "path.sh").write_text("#!/bin/bash")

    # extras that must be excluded
    (recipe / "demo").mkdir()
    (recipe / "demo" / "app.py").write_text("# demo")
    (recipe / "__init__.py").write_text("")
    (recipe / "__pycache__").mkdir()

    return recipe


# ---------------------------------------------------------------------------
# resolver: resolve_recipe
# ---------------------------------------------------------------------------


def test_resolve_recipe_returns_correct_path(fake_egs3):
    path = resolve_recipe("mini_an4/asr")
    assert path == fake_egs3 / "mini_an4" / "asr"


def test_resolve_recipe_raises_value_error_for_no_slash():
    with pytest.raises(ValueError, match="<dataset>/<task>"):
        resolve_recipe("mini_an4")


def test_resolve_recipe_raises_value_error_for_too_many_parts():
    with pytest.raises(ValueError, match="<dataset>/<task>"):
        resolve_recipe("a/b/c")


def test_resolve_recipe_raises_file_not_found_for_unknown_recipe(fake_egs3):
    with pytest.raises(FileNotFoundError, match="unknown/asr"):
        resolve_recipe("unknown/asr")


def test_resolve_recipe_error_message_lists_available_recipes(fake_egs3):
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_recipe("unknown/asr")
    msg = str(exc_info.value)
    assert "librispeech/asr" in msg
    assert "mini_an4/asr" in msg


# ---------------------------------------------------------------------------
# resolver: list_recipes / _list_available
# ---------------------------------------------------------------------------


def test_list_recipes_returns_dataset_task_pairs(fake_egs3):
    assert list_recipes() == ["librispeech/asr", "mini_an4/asr"]


def test_list_recipes_excludes_template(fake_egs3):
    assert not any(r.startswith("TEMPLATE") for r in list_recipes())


def test_list_recipes_excludes_dotdirs(fake_egs3):
    assert not any(r.startswith(".") for r in list_recipes())


def test_list_recipes_excludes_pycache(fake_egs3):
    assert not any("__pycache__" in r for r in list_recipes())


def test_list_recipes_returns_empty_for_empty_egs3(tmp_path):
    assert _list_available(tmp_path) == []


def test_list_recipes_is_sorted(fake_egs3):
    recipes = list_recipes()
    assert recipes == sorted(recipes)


# ---------------------------------------------------------------------------
# command: _copy_recipe
# ---------------------------------------------------------------------------


def test_copy_recipe_creates_destination(fake_recipe, tmp_path):
    dest = tmp_path / "clone"
    _copy_recipe(fake_recipe, dest)
    assert dest.is_dir()


def test_copy_recipe_includes_all_include_items(fake_recipe, tmp_path):
    dest = tmp_path / "clone"
    _copy_recipe(fake_recipe, dest)
    for item in _INCLUDE:
        assert (dest / item).exists(), f"expected {item} in clone"


def test_copy_recipe_excludes_demo(fake_recipe, tmp_path):
    dest = tmp_path / "clone"
    _copy_recipe(fake_recipe, dest)
    assert not (dest / "demo").exists()


def test_copy_recipe_excludes_pycache_inside_subdirs(fake_recipe, tmp_path):
    dest = tmp_path / "clone"
    _copy_recipe(fake_recipe, dest)
    for d in ("conf", "src", "dataset"):
        assert not (dest / d / "__pycache__").exists()


def test_copy_recipe_skips_missing_include_item(tmp_path):
    """Recipe missing some _INCLUDE entries should still copy without error."""
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "run.py").write_text("# run")

    dest = tmp_path / "clone"
    _copy_recipe(sparse, dest)

    assert (dest / "run.py").exists()
    assert not (dest / "conf").exists()


# ---------------------------------------------------------------------------
# command: run (integration — real mini_an4/asr clone)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cloned_mini_an4(tmp_path_factory):
    """Clone the real mini_an4/asr recipe once for the whole test module."""
    dest = tmp_path_factory.mktemp("integration") / "mini_an4_asr"
    args = argparse.Namespace(list=False, recipe="mini_an4/asr", project=str(dest))
    run(args)
    return dest


def test_integration_clone_creates_directory(cloned_mini_an4):
    assert cloned_mini_an4.is_dir()


def test_integration_all_include_items_present(cloned_mini_an4):
    for item in _INCLUDE:
        assert (cloned_mini_an4 / item).exists(), f"missing: {item}"


def test_integration_conf_has_training_yaml(cloned_mini_an4):
    assert (cloned_mini_an4 / "conf" / "training.yaml").exists()


def test_integration_conf_has_inference_yaml(cloned_mini_an4):
    assert (cloned_mini_an4 / "conf" / "inference.yaml").exists()


def test_integration_no_pycache_in_clone(cloned_mini_an4):
    for pycache in cloned_mini_an4.rglob("__pycache__"):
        pytest.fail(f"__pycache__ found in clone: {pycache}")


def test_integration_no_demo_dir_in_clone(cloned_mini_an4):
    assert not (cloned_mini_an4 / "demo").exists()


def test_integration_no_init_py_at_root(cloned_mini_an4):
    assert not (cloned_mini_an4 / "__init__.py").exists()


def test_integration_no_dotfiles_at_root(cloned_mini_an4):
    dotfiles = [p.name for p in cloned_mini_an4.iterdir() if p.name.startswith(".")]
    assert dotfiles == [], f"unexpected dotfiles: {dotfiles}"


def test_integration_publication_yaml_has_hf_repo(cloned_mini_an4):
    text = (cloned_mini_an4 / "conf" / "publication.yaml").read_text(encoding="utf-8")
    assert "hf_repo: espnet/mini_an4_asr_${exp_tag}" in text


def test_integration_demo_yaml_has_title(cloned_mini_an4):
    text = (cloned_mini_an4 / "conf" / "demo.yaml").read_text(encoding="utf-8")
    assert "title: mini_an4_asr demo" in text


def test_integration_demo_yaml_has_upload_hf_repo(cloned_mini_an4):
    text = (cloned_mini_an4 / "conf" / "demo.yaml").read_text(encoding="utf-8")
    assert "hf_repo: espnet/mini_an4_asr_${exp_tag}" in text


def test_integration_run_py_help_exits_cleanly(cloned_mini_an4):
    """python run.py --help must exit 0 and mention --stages."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=cloned_mini_an4,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--stages" in result.stdout


# ---------------------------------------------------------------------------
# command: run — error cases (fake_egs3 sufficient)
# ---------------------------------------------------------------------------


def test_run_raises_file_exists_error_when_dest_exists(fake_egs3, tmp_path):
    dest = tmp_path / "existing"
    dest.mkdir()
    args = argparse.Namespace(list=False, recipe="mini_an4/asr", project=str(dest))
    with pytest.raises(FileExistsError, match="already exists"):
        run(args)


def test_run_raises_file_not_found_for_unknown_recipe(fake_egs3, tmp_path):
    dest = tmp_path / "my_project"
    args = argparse.Namespace(list=False, recipe="unknown/asr", project=str(dest))
    with pytest.raises(FileNotFoundError):
        run(args)


def test_run_does_not_create_directory_on_unknown_recipe(fake_egs3, tmp_path):
    dest = tmp_path / "my_project"
    args = argparse.Namespace(list=False, recipe="unknown/asr", project=str(dest))
    with pytest.raises(FileNotFoundError):
        run(args)
    assert not dest.exists()


# ---------------------------------------------------------------------------
# command: run — additional error cases
# ---------------------------------------------------------------------------


def test_run_raises_file_exists_error_when_dest_is_a_file(fake_egs3, tmp_path):
    dest = tmp_path / "existing_file"
    dest.write_text("not a directory")
    args = argparse.Namespace(list=False, recipe="mini_an4/asr", project=str(dest))
    with pytest.raises(FileExistsError, match="already exists"):
        run(args)


def test_run_creates_nested_project_path(fake_egs3, tmp_path):
    """--project a/b/c should succeed even when parents do not exist."""
    dest = tmp_path / "a" / "b" / "c"
    args = argparse.Namespace(list=False, recipe="mini_an4/asr", project=str(dest))
    run(args)
    assert dest.is_dir()


def test_run_raises_value_error_for_bad_recipe_format(fake_egs3, tmp_path):
    dest = tmp_path / "my_project"
    args = argparse.Namespace(list=False, recipe="no_slash", project=str(dest))
    with pytest.raises(ValueError, match="<dataset>/<task>"):
        run(args)


def test_run_does_not_create_directory_on_bad_recipe_format(fake_egs3, tmp_path):
    dest = tmp_path / "my_project"
    args = argparse.Namespace(list=False, recipe="no_slash", project=str(dest))
    with pytest.raises(ValueError):
        run(args)
    assert not dest.exists()


# ---------------------------------------------------------------------------
# resolver: resolve_recipe — additional edge cases
# ---------------------------------------------------------------------------


def test_resolve_recipe_raises_for_empty_string():
    with pytest.raises(ValueError, match="<dataset>/<task>"):
        resolve_recipe("")


def test_resolve_recipe_raises_for_only_slash():
    with pytest.raises(ValueError, match="<dataset>/<task>"):
        resolve_recipe("/")


def test_resolve_recipe_raises_for_empty_task(fake_egs3):
    """'mini_an4/' strips to 'mini_an4' (one part) → ValueError."""
    with pytest.raises(ValueError, match="<dataset>/<task>"):
        resolve_recipe("mini_an4/")


def test_resolve_recipe_accepts_surrounding_slashes(fake_egs3):
    """/mini_an4/asr/ should resolve the same as mini_an4/asr."""
    path = resolve_recipe("/mini_an4/asr/")
    assert path == fake_egs3 / "mini_an4" / "asr"


def test_resolve_recipe_dataset_exists_but_task_missing(fake_egs3):
    """`mini_an4` dir exists but `mini_an4/tts` does not → FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="mini_an4/tts"):
        resolve_recipe("mini_an4/tts")


def test_resolve_recipe_error_shows_available_when_dataset_missing(fake_egs3):
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_recipe("nosuchdataset/asr")
    msg = str(exc_info.value)
    assert "mini_an4/asr" in msg
    assert "librispeech/asr" in msg


# ---------------------------------------------------------------------------
# resolver: _list_available — additional edge cases
# ---------------------------------------------------------------------------


def test_list_available_skips_files_in_egs3_root(tmp_path):
    (tmp_path / "mini_an4" / "asr").mkdir(parents=True)
    (tmp_path / "not_a_dir.txt").write_text("file")
    result = _list_available(tmp_path)
    assert "not_a_dir.txt" not in result
    assert "mini_an4/asr" in result


def test_list_available_skips_dataset_with_no_task_subdirs(tmp_path):
    """A dataset dir that has no subdirectories contributes nothing."""
    (tmp_path / "empty_dataset").mkdir()
    (tmp_path / "mini_an4" / "asr").mkdir(parents=True)
    result = _list_available(tmp_path)
    assert not any(r.startswith("empty_dataset") for r in result)
    assert "mini_an4/asr" in result


def test_list_available_skips_underscore_prefixed_task_dirs(tmp_path):
    (tmp_path / "mini_an4" / "_private").mkdir(parents=True)
    (tmp_path / "mini_an4" / "asr").mkdir(parents=True)
    result = _list_available(tmp_path)
    assert "mini_an4/_private" not in result
    assert "mini_an4/asr" in result


def test_list_available_skips_task_files_not_dirs(tmp_path):
    (tmp_path / "mini_an4").mkdir()
    (tmp_path / "mini_an4" / "asr").mkdir()
    (tmp_path / "mini_an4" / "README.md").write_text("docs")
    result = _list_available(tmp_path)
    assert "mini_an4/README.md" not in result
    assert "mini_an4/asr" in result


# ---------------------------------------------------------------------------
# command: _copy_recipe — additional edge cases
# ---------------------------------------------------------------------------


def test_copy_recipe_preserves_file_content(tmp_path):
    src = tmp_path / "recipe"
    src.mkdir()
    (src / "run.py").write_text("print('hello')")

    dest = tmp_path / "clone"
    _copy_recipe(src, dest)

    assert (dest / "run.py").read_text() == "print('hello')"


def test_copy_recipe_empty_recipe_creates_empty_dest(tmp_path):
    """A recipe with none of the _INCLUDE items produces an empty directory."""
    src = tmp_path / "recipe"
    src.mkdir()
    (src / "demo").mkdir()
    (src / "__init__.py").write_text("")

    dest = tmp_path / "clone"
    _copy_recipe(src, dest)

    assert dest.is_dir()
    assert list(dest.iterdir()) == []


def test_inject_corpus_system_publication_yaml(tmp_path):
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "publication.yaml").write_text("pack_model:\n  include: []\n")
    _inject_corpus_system(tmp_path, "mini_an4/asr")
    text = (conf / "publication.yaml").read_text(encoding="utf-8")
    assert "hf_repo: espnet/mini_an4_asr_${exp_tag}" in text


def test_inject_corpus_system_demo_yaml(tmp_path):
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "demo.yaml").write_text("model:\n  trust_user_code: true\n")
    _inject_corpus_system(tmp_path, "mini_an4/asr")
    text = (conf / "demo.yaml").read_text(encoding="utf-8")
    assert "title: mini_an4_asr demo" in text
    assert "hf_repo: espnet/mini_an4_asr_${exp_tag}" in text


def test_inject_corpus_system_skips_missing_conf_files(tmp_path):
    """Should not crash when conf files are absent."""
    (tmp_path / "conf").mkdir()
    _inject_corpus_system(tmp_path, "mini_an4/asr")  # no publication.yaml or demo.yaml


def test_inject_corpus_system_no_conf_dir(tmp_path):
    """Should not crash when conf/ does not exist at all."""
    _inject_corpus_system(tmp_path, "mini_an4/asr")


# ---------------------------------------------------------------------------
# command: add_arguments
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# command: --list flag
# ---------------------------------------------------------------------------


def test_clone_list_prints_available_recipes(fake_egs3, capsys):
    args = argparse.Namespace(list=True, recipe=None, project=None)
    run(args)
    out = capsys.readouterr().out
    assert "mini_an4/asr" in out
    assert "librispeech/asr" in out


def test_clone_list_does_not_require_recipe(fake_egs3, capsys):
    args = argparse.Namespace(list=True, recipe=None, project=None)
    run(args)  # must not raise


def test_clone_list_prints_no_recipes_when_egs3_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(resolver_module, "_get_egs3_root", lambda: tmp_path)
    args = argparse.Namespace(list=True, recipe=None, project=None)
    run(args)
    out = capsys.readouterr().out
    assert "No recipes available" in out


def test_clone_list_does_not_clone_anything(fake_egs3, tmp_path, monkeypatch, capsys):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    args = argparse.Namespace(list=True, recipe="mini_an4/asr", project=None)
    run(args)
    assert not (workdir / "mini_an4").exists()


# ---------------------------------------------------------------------------
# command: add_arguments
# ---------------------------------------------------------------------------


def test_add_arguments_registers_clone_subcommand():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_arguments(sub)
    args = parser.parse_args(["clone", "mini_an4/asr", "--project", "proj"])
    assert args.recipe == "mini_an4/asr"
    assert args.project == "proj"
    assert args.func is run


def test_add_arguments_project_defaults_to_none():
    """--project is optional; None triggers default-path logic in run()."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_arguments(sub)
    args = parser.parse_args(["clone", "mini_an4/asr"])
    assert args.project is None


def test_add_arguments_recipe_is_none_when_omitted():
    """recipe is optional at parse time; run() validates it."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_arguments(sub)
    args = parser.parse_args(["clone", "--project", "proj"])
    assert args.recipe is None


def test_run_uses_recipe_as_default_dest(fake_egs3, tmp_path, monkeypatch):
    """No --project → dest is cwd/<recipe>."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    args = argparse.Namespace(list=False, recipe="mini_an4/asr", project=None)
    run(args)
    assert (workdir / "mini_an4" / "asr").is_dir()


def test_run_default_dest_raises_if_already_exists(fake_egs3, tmp_path, monkeypatch):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "mini_an4" / "asr").mkdir(parents=True)
    monkeypatch.chdir(workdir)
    args = argparse.Namespace(list=False, recipe="mini_an4/asr", project=None)
    with pytest.raises(FileExistsError):
        run(args)


def test_run_raises_value_error_when_recipe_is_none(tmp_path):
    args = argparse.Namespace(list=False, recipe=None, project=str(tmp_path / "proj"))
    with pytest.raises(ValueError, match="recipe argument is required"):
        run(args)


def test_run_missing_recipe_error_shows_usage(tmp_path):
    args = argparse.Namespace(list=False, recipe=None, project=str(tmp_path / "proj"))
    with pytest.raises(ValueError) as exc_info:
        run(args)
    msg = str(exc_info.value)
    assert "espnet3 clone <dataset>/<task>" in msg
    assert "espnet3 clone --help" in msg
    assert "mini_an4/asr" in msg


def test_run_missing_recipe_does_not_create_directory(tmp_path):
    dest = tmp_path / "proj"
    args = argparse.Namespace(list=False, recipe=None, project=str(dest))
    with pytest.raises(ValueError):
        run(args)
    assert not dest.exists()
