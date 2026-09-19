"""Tests for :mod:`espnet3.cli.clone.command`."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pytest

from espnet3.cli.clone.command import (
    _INCLUDE,
    _copy_recipe,
    _inject_corpus_system,
    add_arguments,
    run,
)


def test_copy_recipe_creates_destination_and_includes_items(fake_recipe, tmp_path):
    dest = tmp_path / "clone"
    _copy_recipe(fake_recipe, dest)
    assert dest.is_dir()
    assert all((dest / item).exists() for item in _INCLUDE)


def test_copy_recipe_excludes_unwanted_files(fake_recipe, tmp_path):
    dest = tmp_path / "clone"
    _copy_recipe(fake_recipe, dest)
    assert not (dest / "demo").exists()
    assert not (dest / "__init__.py").exists()
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
    (empty / "demo").mkdir()
    empty_dest = tmp_path / "empty_clone"
    _copy_recipe(empty, empty_dest)
    assert list(empty_dest.iterdir()) == []


def test_copy_recipe_normalizes_internal_symlink_targets(tmp_path):
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
        assert link.is_symlink() and not os.path.isabs(os.readlink(link))
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


def test_inject_corpus_system_adds_missing_fields(tmp_path):
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    (conf_dir / "publication.yaml").write_text("pack_model:\n  include: []\n")
    (conf_dir / "demo.yaml").write_text("model:\n  trust_user_code: true\n")
    _inject_corpus_system(tmp_path, "mini_an4/asr")
    publication = (conf_dir / "publication.yaml").read_text()
    demo = (conf_dir / "demo.yaml").read_text()
    assert "hf_repo: espnet/mini_an4_asr_${exp_tag}" in publication
    assert "title: mini_an4_asr demo" in demo
    assert "hf_repo: espnet/mini_an4_asr_${exp_tag}" in demo


def test_inject_corpus_system_updates_existing_keys_without_duplicates(tmp_path):
    from omegaconf import OmegaConf

    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    (conf_dir / "publication.yaml").write_text(
        "upload_model:\n  hf_repo: existing/model\n  private: false\n"
    )
    (conf_dir / "demo.yaml").write_text(
        "ui:\n  title: Existing demo\n  app_script: src/app.py\n"
        "upload_demo:\n  hf_repo: existing/demo\n  update: false\n"
    )
    _inject_corpus_system(tmp_path, "mini_an4/asr")
    OmegaConf.load(conf_dir / "publication.yaml")
    OmegaConf.load(conf_dir / "demo.yaml")
    publication = (conf_dir / "publication.yaml").read_text()
    demo = (conf_dir / "demo.yaml").read_text()
    assert publication.count("upload_model:") == 1 and "private: false" in publication
    assert demo.count("ui:") == demo.count("upload_demo:") == 1
    assert "app_script: src/app.py" in demo and "update: false" in demo


def test_inject_corpus_system_skips_missing_config_files(tmp_path):
    (tmp_path / "conf").mkdir()
    _inject_corpus_system(tmp_path, "mini_an4/asr")
    _inject_corpus_system(tmp_path / "no_conf", "mini_an4/asr")


@pytest.fixture(scope="module")
def cloned_mini_an4(tmp_path_factory):
    dest = tmp_path_factory.mktemp("integration") / "mini_an4_asr"
    run(argparse.Namespace(list=False, recipe="mini_an4/asr", project=str(dest)))
    return dest


def test_integration_clone_layout(cloned_mini_an4):
    assert cloned_mini_an4.is_dir()
    assert all((cloned_mini_an4 / item).exists() for item in _INCLUDE)
    assert (cloned_mini_an4 / "conf" / "training.yaml").exists()
    assert (cloned_mini_an4 / "conf" / "inference.yaml").exists()
    assert not any(cloned_mini_an4.rglob("__pycache__"))
    assert not (cloned_mini_an4 / "demo").exists()
    assert not (cloned_mini_an4 / "__init__.py").exists()
    assert not [p for p in cloned_mini_an4.iterdir() if p.name.startswith(".")]


def test_integration_clone_injects_recipe_identity(cloned_mini_an4):
    publication = (cloned_mini_an4 / "conf" / "publication.yaml").read_text()
    demo = (cloned_mini_an4 / "conf" / "demo.yaml").read_text()
    assert "hf_repo: espnet/mini_an4_asr_${exp_tag}" in publication
    assert "title: mini_an4_asr demo" in demo
    assert "hf_repo: espnet/mini_an4_asr_${exp_tag}" in demo


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


@pytest.mark.parametrize("recipe", ("unknown/asr", "no_slash"))
def test_run_rejects_unknown_or_invalid_recipe(fake_egs3, tmp_path, recipe):
    dest = tmp_path / "project"
    with pytest.raises((FileNotFoundError, ValueError)):
        run(argparse.Namespace(list=False, recipe=recipe, project=str(dest)))
    assert not dest.exists()


def test_run_rejects_existing_destination(fake_egs3, tmp_path):
    dest = tmp_path / "existing"
    dest.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run(argparse.Namespace(list=False, recipe="mini_an4/asr", project=str(dest)))
    file_dest = tmp_path / "existing_file"
    file_dest.write_text("not a directory")
    with pytest.raises(FileExistsError, match="already exists"):
        run(
            argparse.Namespace(
                list=False, recipe="mini_an4/asr", project=str(file_dest)
            )
        )


def test_run_creates_explicit_and_default_destination(fake_egs3, tmp_path, monkeypatch):
    explicit = tmp_path / "a" / "b" / "c"
    run(argparse.Namespace(list=False, recipe="mini_an4/asr", project=str(explicit)))
    assert explicit.is_dir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    run(argparse.Namespace(list=False, recipe="mini_an4/asr", project=None))
    assert (workdir / "mini_an4" / "asr").is_dir()


def test_run_rejects_existing_default_destination(fake_egs3, tmp_path, monkeypatch):
    workdir = tmp_path / "workdir"
    (workdir / "mini_an4" / "asr").mkdir(parents=True)
    monkeypatch.chdir(workdir)
    with pytest.raises(FileExistsError):
        run(argparse.Namespace(list=False, recipe="mini_an4/asr", project=None))


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
    assert "mini_an4/asr" in output and "librispeech/asr" in output


def test_clone_list_handles_empty_and_does_not_clone(
    empty_egs3, tmp_path, monkeypatch, capsys
):
    run(argparse.Namespace(list=True, recipe=None, project=None))
    assert "No recipes available" in capsys.readouterr().out
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    run(argparse.Namespace(list=True, recipe="mini_an4/asr", project=None))
    assert not (workdir / "mini_an4").exists()


def test_add_arguments_registers_clone_subcommand():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_arguments(subparsers)
    args = parser.parse_args(["clone", "mini_an4/asr", "--project", "project"])
    assert args.recipe == "mini_an4/asr" and args.project == "project"
    assert args.func is run
    args = parser.parse_args(["clone", "mini_an4/asr"])
    assert args.project is None
    args = parser.parse_args(["clone", "--project", "project"])
    assert args.recipe is None
