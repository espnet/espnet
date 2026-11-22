"""The rule that decides what a release is called.

It is load-bearing: the milestone title becomes version.txt, version.txt is
what PyPI publishes, and the tag is `v.` plus the same string. A title the
rule lets through and PyPI will not accept is a release that dies at the
upload, which is how the 202604 series went out wrong.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

SCRIPT = Path(__file__).parents[2] / "doc" / "make_release.py"


def _module():
    # doc/ is not a package, and the script imports github at module level.
    # PyGithub is not an espnet dependency, so CI does not have it. None of
    # the functions tested here talk to GitHub, so an empty stand-in lets the
    # script load; skipping instead would leave every test here silently
    # unrun. Anything that does reach for GitHub fails with AttributeError.
    try:
        import github  # noqa: F401
    except ImportError:
        stub = {"github": types.ModuleType("github")}
    else:
        stub = {}
    spec = importlib.util.spec_from_file_location("make_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stub):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "title, version",
    [
        ("v.202610", "202610"),
        ("v.202610.post1", "202610.post1"),
        ("v.202610.post12", "202610.post12"),
        ("202610", "202610"),  # the prefix is optional
    ],
)
def test_a_release_milestone_names_its_version(title, version):
    assert _module().release_version(title) == version


@pytest.mark.parametrize(
    "title",
    [
        # PyPI will not take this one, and version.txt is what it publishes
        "v.202610-patch1",
        "v.202610patch1",
        "v.2026101",
        "v.20261",
        "v.202610.post",
        "v.202610.dev1",
        # \d would take these; PyPI would not
        "v.２０２６１０",
        "v.202610.post１",
        "Backlog",
        "",
    ],
)
def test_anything_pypi_would_refuse_is_not_a_release(title):
    assert _module().release_version(title) is None


def _pin_files(module, tmp_path, monkeypatch, dockerfile, readme):
    (tmp_path / "Dockerfile").write_text(dockerfile)
    (tmp_path / "README.md").write_text(readme)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "VERSION_PINS",
        [(tmp_path / "Dockerfile", pattern) for _, pattern in module.VERSION_PINS[:1]]
        + [(tmp_path / "README.md", pattern) for _, pattern in module.VERSION_PINS[1:]],
    )


def test_apply_writes_the_release_into_the_docker_pins(tmp_path, monkeypatch):
    module = _module()
    _pin_files(
        module,
        tmp_path,
        monkeypatch,
        "FROM x\nARG ESPNET_VERSION=202610\nRUN pip install espnet\n",
        "docker build --build-arg ESPNET_VERSION=202610 -t espnet-inference .\n",
    )
    assert module.pins_up_to_date("202611", will_apply=False)
    # --apply is what writes them, so their being stale is not a problem there
    assert module.pins_up_to_date("202611", will_apply=True) == []

    module.update_pins("202611")

    assert (tmp_path / "Dockerfile").read_text() == (
        "FROM x\nARG ESPNET_VERSION=202611\nRUN pip install espnet\n"
    )
    assert (
        "--build-arg ESPNET_VERSION=202611 -t" in (tmp_path / "README.md").read_text()
    )
    assert module.pins_up_to_date("202611", will_apply=False) == []


def test_a_pin_the_pattern_no_longer_finds_fails_even_under_apply(
    tmp_path, monkeypatch
):
    module = _module()
    _pin_files(module, tmp_path, monkeypatch, "FROM x\n", "nothing here\n")
    assert len(module.pins_up_to_date("202611", will_apply=True)) == 2


def test_the_docker_pins_name_the_release_in_version_txt():
    module = _module()
    version = module.VERSION_FILE.read_text().strip()
    for path, versions in module.pinned_versions():
        assert versions, f"{path}: VERSION_PINS finds nothing"
        assert set(versions) == {version}, f"{path} says {versions}, not {version}"
