"""Tests for :mod:`espnet3.bin.clone.command`."""

from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

from espnet3.bin.clone.command import (
    _copy_recipe,
    add_arguments,
    run,
)


def test_copy_recipe_copies_complete_recipe_package(fake_recipe, tmp_path):
    dest = tmp_path / "clone"
    _copy_recipe(fake_recipe, dest)
    assert dest.is_dir()
    assert all(
        (dest / item).exists()
        for item in (
            "conf",
            "src",
            "dataset",
            "run.py",
            "readme.md",
            "path.sh",
            "__init__.py",
        )
    )
    assert (dest / "assets" / "metadata.json").read_text() == '{"version": 1}'


def test_copy_recipe_excludes_only_gitignored_artifacts(fake_recipe, tmp_path):
    dest = tmp_path / "clone"
    _copy_recipe(fake_recipe, dest)
    for name in ("data", "exp", "dump", "downloads", ".cache", ".pytest_cache"):
        assert not (dest / name).exists()
    assert not (dest / "sample.pyc").exists()
    assert (dest / "demo" / "app.py").is_file()
    assert (dest / "downloads.tar.gz").read_bytes() == b""
    assert (dest / ".agents" / "instructions.md").read_text() == "ignore"
    assert (dest / "dataset" / "file.py").is_file()
    assert (dest / "dataset" / "data" / "README.md").read_text() == "recipe source"
    assert not any(dest.rglob("__pycache__"))


def test_copy_recipe_skips_missing_include_item(tmp_path):
    source = tmp_path / "sparse"
    source.mkdir()
    (source / "run.py").write_text("# run")
    dest = tmp_path / "clone"
    _copy_recipe(source, dest)
    assert (dest / "run.py").exists()
    assert not (dest / "conf").exists()


def test_copy_recipe_preserves_file_content_and_empty_recipe(tmp_path):
    source = tmp_path / "recipe"
    source.mkdir()
    (source / "run.py").write_text("print('hello')")
    dest = tmp_path / "clone"
    _copy_recipe(source, dest)
    assert (dest / "run.py").read_text() == "print('hello')"

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "__pycache__").mkdir()
    empty_dest = tmp_path / "empty_clone"
    _copy_recipe(empty, empty_dest)
    assert list(empty_dest.iterdir()) == []


def test_copy_recipe_dereferences_internal_symlink_targets(tmp_path):
    source = tmp_path / "recipe"
    conf = source / "conf"
    conf.mkdir(parents=True)
    target = conf / "training_actual.yaml"
    target.write_text("config: true")
    (conf / "relative.yaml").symlink_to(target.name)
    (conf / "absolute.yaml").symlink_to(target)
    dest = tmp_path / "clone"
    _copy_recipe(source, dest)
    for name in ("relative.yaml", "absolute.yaml"):
        link = dest / "conf" / name
        assert link.is_file() and not link.is_symlink()
        assert link.read_text() == "config: true"


def test_copy_recipe_dereferences_external_symlink(tmp_path):
    external = tmp_path / "external.yaml"
    external.write_text("external: true")
    source = tmp_path / "recipe"
    (source / "conf").mkdir(parents=True)
    (source / "conf" / "external.yaml").symlink_to(external)
    dest = tmp_path / "clone"
    _copy_recipe(source, dest)
    copied = dest / "conf" / "external.yaml"
    assert copied.is_file() and not copied.is_symlink()
    assert copied.read_text() == "external: true"


@pytest.fixture(scope="module")
def cloned_mini_an4(tmp_path_factory):
    dest = tmp_path_factory.mktemp("integration") / "mini_an4_esp2_asr"
    run(argparse.Namespace(list=False, recipe="mini_an4/esp2_asr", project=str(dest)))
    return dest


def test_integration_clone_layout(cloned_mini_an4):
    assert cloned_mini_an4.is_dir()
    assert (cloned_mini_an4 / "__init__.py").exists()
    assert (cloned_mini_an4 / "conf" / "training.yaml").exists()
    assert (cloned_mini_an4 / "conf" / "inference.yaml").exists()
    assert (cloned_mini_an4 / "downloads.tar.gz").is_file()
    assert not any(cloned_mini_an4.rglob("__pycache__"))


def test_integration_clone_preserves_recipe_configs(cloned_mini_an4):
    from espnet3.bin.clone.resolver import resolve_recipe

    source = resolve_recipe("mini_an4/esp2_asr")
    for config in ("publication.yaml", "demo.yaml"):
        source_config = source / "conf" / config
        if source_config.exists():
            assert (cloned_mini_an4 / "conf" / config).read_bytes() == (
                source_config.read_bytes()
            )


def test_integration_clone_preserves_path_sh(cloned_mini_an4):
    from espnet3.bin.clone.resolver import resolve_recipe

    source = resolve_recipe("mini_an4/esp2_asr") / "path.sh"
    assert (cloned_mini_an4 / "path.sh").read_bytes() == source.read_bytes()


@pytest.mark.execution_timeout(30)
def test_integration_run_py_help_exits_cleanly(cloned_mini_an4):
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=cloned_mini_an4,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--stages" in result.stdout


@pytest.mark.parametrize("recipe", ("unknown/esp2_asr", "no_slash"))
def test_run_rejects_unknown_or_invalid_recipe(fake_egs3, tmp_path, recipe):
    dest = tmp_path / "project"
    with pytest.raises((FileNotFoundError, ValueError)):
        run(argparse.Namespace(list=False, recipe=recipe, project=str(dest)))
    assert not dest.exists()


def test_run_rejects_existing_destination(fake_egs3, tmp_path):
    dest = tmp_path / "existing"
    dest.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run(
            argparse.Namespace(
                list=False, recipe="mini_an4/esp2_asr", project=str(dest)
            )
        )
    file_dest = tmp_path / "existing_file"
    file_dest.write_text("not a directory")
    with pytest.raises(FileExistsError, match="already exists"):
        run(
            argparse.Namespace(
                list=False, recipe="mini_an4/esp2_asr", project=str(file_dest)
            )
        )


def test_run_creates_explicit_and_default_destination(fake_egs3, tmp_path, monkeypatch):
    explicit = tmp_path / "a" / "b" / "c"
    run(
        argparse.Namespace(
            list=False, recipe="mini_an4/esp2_asr", project=str(explicit)
        )
    )
    assert explicit.is_dir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    run(argparse.Namespace(list=False, recipe="mini_an4/esp2_asr", project=None))
    assert (workdir / "mini_an4" / "esp2_asr").is_dir()


def test_default_destination_strips_recipe_slashes(fake_egs3, tmp_path, monkeypatch):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    run(argparse.Namespace(list=False, recipe="/mini_an4/esp2_asr/", project=None))
    assert (workdir / "mini_an4" / "esp2_asr").is_dir()


def test_run_rejects_existing_default_destination(fake_egs3, tmp_path, monkeypatch):
    workdir = tmp_path / "workdir"
    (workdir / "mini_an4" / "esp2_asr").mkdir(parents=True)
    monkeypatch.chdir(workdir)
    with pytest.raises(FileExistsError):
        run(argparse.Namespace(list=False, recipe="mini_an4/esp2_asr", project=None))


def test_run_rejects_missing_recipe(tmp_path):
    args = argparse.Namespace(
        list=False, recipe=None, project=str(tmp_path / "project")
    )
    with pytest.raises(ValueError, match="recipe argument is required") as error:
        run(args)
    assert "espnet3 clone <dataset>/<task>" in str(error.value)
    assert "espnet3 clone --help" in str(error.value)
    assert not (tmp_path / "project").exists()


def test_clone_list_uses_resolver_output(fake_egs3, capsys):
    run(argparse.Namespace(list=True, recipe=None, project=None))
    output = capsys.readouterr().out
    assert "mini_an4/esp2_asr" in output and "librispeech/esp2_asr" in output


def test_espnet3_main_clones_recipe(fake_egs3, tmp_path, capsys):
    from espnet3.bin.main import main

    project = tmp_path / "project"
    assert main(["clone", "mini_an4/esp2_asr", "--project", str(project)]) == 0
    assert project.is_dir()


def test_espnet3_main_reports_user_errors_without_traceback(fake_egs3, capsys):
    from espnet3.bin.main import main

    assert main(["clone", "unknown/esp2_asr"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("espnet3:")
    assert "Traceback" not in captured.err


def test_clone_list_handles_empty_and_does_not_clone(
    empty_egs3, tmp_path, monkeypatch, capsys
):
    run(argparse.Namespace(list=True, recipe=None, project=None))
    assert "No recipes available" in capsys.readouterr().out
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    run(argparse.Namespace(list=True, recipe="mini_an4/esp2_asr", project=None))
    assert not (workdir / "mini_an4").exists()


def test_add_arguments_registers_clone_subcommand():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_arguments(subparsers)
    args = parser.parse_args(["clone", "mini_an4/esp2_asr", "--project", "project"])
    assert args.recipe == "mini_an4/esp2_asr" and args.project == "project"
    assert args.func is run
    args = parser.parse_args(["clone", "mini_an4/esp2_asr"])
    assert args.project is None
    args = parser.parse_args(["clone", "--project", "project"])
    assert args.recipe is None
