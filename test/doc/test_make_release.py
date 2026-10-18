"""The rule that decides what a release is called.

It is load-bearing: the milestone title becomes version.txt, version.txt is
what PyPI publishes, and the tag is `v.` plus the same string. A title the
rule lets through and PyPI will not accept is a release that dies at the
upload, which is how the 202604 series went out wrong.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "doc" / "make_release.py"


def _module():
    # doc/ is not a package, and the script imports github at module level
    pytest.importorskip("github")
    spec = importlib.util.spec_from_file_location("make_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
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
