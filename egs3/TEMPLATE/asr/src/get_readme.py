from pathlib import Path


def get_readme() -> str:
    """Return the pack-model README template text."""
    return (Path(__file__).parent / "readme.md").read_text(encoding="utf-8")


def get_demo_readme() -> str:
    """Return the demo README template text."""
    return (Path(__file__).parent / "demo_readme.md").read_text(encoding="utf-8")
