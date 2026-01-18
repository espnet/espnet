"""Demo asset setup helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from omegaconf import OmegaConf

_APP_TEMPLATE = """
from pathlib import Path

from espnet3.demo.app_builder import build_demo_app

DEMO_DIR = Path(__file__).resolve().parent


def main():
    app = build_demo_app(DEMO_DIR)
    app.launch()


if __name__ == "__main__":
    main()
"""


def setup_demo_assets(
    *,
    demo_dir: Path,
    demo_config,
    requirements: Iterable[str] | None = None,
    ui_templates: Dict[str, dict] | None = None,
) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    _write_app_py(demo_dir)
    _write_requirements(demo_dir, demo_config, requirements)
    if ui_templates:
        _write_ui_templates(demo_dir, ui_templates)


def _write_app_py(demo_dir: Path) -> None:
    (demo_dir / "app.py").write_text(_APP_TEMPLATE, encoding="utf-8")


def _write_requirements(
    demo_dir: Path,
    demo_config,
    requirements: Iterable[str] | None,
) -> None:
    reqs = list(requirements or [])
    if not reqs:
        cfg_reqs = getattr(demo_config, "requirements", None)
        if cfg_reqs:
            reqs = list(cfg_reqs)
    if not reqs:
        reqs = ["gradio"]
    (demo_dir / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")


def _write_ui_templates(demo_dir: Path, templates: Dict[str, dict]) -> None:
    out_dir = demo_dir / "ui_templates"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in templates.items():
        path = out_dir / f"{name}.yaml"
        path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
