"""Check recipe dependency diagnostics without importing optional packages."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[3] / "egs2/geolid/lid1"
DEPENDENCIES = ("s3prl", "transformers", "lang2vec", "datasets", "pycountry")
pytestmark = pytest.mark.skipif(os.name == "nt", reason="Recipes use POSIX scripts")


@pytest.fixture
def dependency_environment(tmp_path, monkeypatch):
    """Shadow optional packages and start each check in a fresh Python process."""
    modules = tmp_path / "modules"
    modules.mkdir()
    for name in DEPENDENCIES:
        (modules / f"{name}.py").write_text("")
    (modules / "datasets.py").write_text('__version__ = "3.6.0"\n')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    executable = binaries / "python3"
    executable.write_text(
        f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PYTHONPATH", str(modules))
    monkeypatch.delenv("MAIN_ROOT", raising=False)
    return modules


def source_dependencies(command=". local/path.sh"):
    return subprocess.run(
        ["bash", "-c", command],
        cwd=RECIPE,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("version", ["2.16.0", "3.6.0", "v3.6.0", "3.6.0+local"])
def test_dependencies_available(dependency_environment, version):
    (dependency_environment / "datasets.py").write_text(f"__version__ = {version!r}\n")

    result = source_dependencies()

    assert result.returncode == 0
    assert result.stdout == result.stderr == ""


@pytest.mark.parametrize("module", DEPENDENCIES)
def test_missing_dependency(dependency_environment, module):
    (dependency_environment / f"{module}.py").write_text(
        f'raise ModuleNotFoundError("No module named {module}")\n'
    )

    result = source_dependencies()

    assert result.returncode == 1
    assert result.stdout == ""
    assert module in result.stderr
    assert "Traceback" not in result.stderr
    if module in ("s3prl", "transformers", "lang2vec"):
        assert f'cd "../../../tools" && ./installers/install_{module}.sh' in (
            result.stderr
        )
    elif module == "datasets":
        assert "python3 -m pip install 'datasets>=2.16,<4'" in result.stderr
    else:
        assert "python3 -m pip install pycountry" in result.stderr


@pytest.mark.parametrize(
    "version", ["4", "4.0.0", "4.0.0rc1", "10.0.0", "unknown", None]
)
def test_incompatible_datasets(dependency_environment, version):
    (dependency_environment / "datasets.py").write_text(f"__version__ = {version!r}\n")

    result = source_dependencies()

    assert result.returncode == 1
    assert result.stdout == ""
    assert "datasets<4" in result.stderr
    assert "python3 -m pip install 'datasets>=2.16,<4'" in result.stderr
    assert "Traceback" not in result.stderr


def test_import_failure_returns_to_sourcing_shell(dependency_environment, monkeypatch):
    (dependency_environment / "s3prl.py").write_text(
        'print("package initialization output")\n'
        'raise RuntimeError("broken package installation")\n'
    )
    monkeypatch.setenv("MAIN_ROOT", "/checkout with spaces/espnet")

    result = source_dependencies(
        '. local/path.sh; status=$?; echo "returned ${status}"'
    )

    assert result.returncode == 0
    assert result.stdout == "returned 1\n"
    assert 'cd "/checkout with spaces/espnet/tools"' in result.stderr
    assert "Traceback" not in result.stderr
    assert "broken package installation" not in result.stderr


def test_dependency_failure_stops_recipe_with_errexit(dependency_environment):
    (dependency_environment / "pycountry.py").write_text(
        'raise ImportError("broken package installation")\n'
    )

    result = source_dependencies("set -e; . local/path.sh; echo should-not-run")

    assert result.returncode == 1
    assert result.stdout == ""
    assert "pycountry" in result.stderr
