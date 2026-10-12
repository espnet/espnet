"""Tests for the layout of espnet3/systems/<name>/."""

from pathlib import Path

import pytest

import espnet3.systems

SYSTEMS_DIR = Path(espnet3.systems.__file__).parent

ALLOWED = frozenset(
    {"__init__.py", "inference.py", "metrics", "system.py", "task.py", "src"}
)
IGNORED = frozenset({"__pycache__"})
EXEMPT = frozenset({"base"})


def _systems() -> list[Path]:
    return sorted(
        p
        for p in SYSTEMS_DIR.iterdir()
        if p.is_dir() and p.name not in EXEMPT and not p.name.startswith(("_", "."))
    )


def _violations(system_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in system_dir.iterdir()
        if p.name not in ALLOWED and p.name not in IGNORED
    )


@pytest.mark.parametrize("system_dir", _systems(), ids=lambda p: p.name)
def test_every_system_keeps_the_layout(system_dir: Path) -> None:
    """Verify every system directory maintains the allowed layout."""
    names = _violations(system_dir)
    assert names == [], f"{system_dir.name}: move under src/: {names}"


def test_extra_entries_are_reported(tmp_path: Path) -> None:
    """Verify that extra entries are reported as violations."""
    system_dir = tmp_path / "demo"
    system_dir.mkdir()
    (system_dir / "inference.py").touch()
    (system_dir / "src").mkdir()
    (system_dir / "__pycache__").mkdir()
    (system_dir / "helpers").mkdir()
    (system_dir / "extra.py").touch()

    assert _violations(system_dir) == ["extra.py", "helpers"]


def test_allowed_entries_pass(tmp_path: Path) -> None:
    """Verify that allowed entries do not cause violations."""
    system_dir = tmp_path / "demo"
    system_dir.mkdir()
    (system_dir / "__init__.py").touch()
    (system_dir / "inference.py").touch()
    (system_dir / "metrics").mkdir()
    (system_dir / "system.py").touch()
    (system_dir / "task.py").touch()
    (system_dir / "src").mkdir()
    (system_dir / "__pycache__").mkdir()

    assert _violations(system_dir) == []


def test_base_is_exempt() -> None:
    """Verify that the base system directory is exempt from checks."""
    systems = _systems()
    assert SYSTEMS_DIR / "base" not in systems
    assert SYSTEMS_DIR / "esp2_asr" in systems
    assert SYSTEMS_DIR / "f5tts" in systems
