from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.demo.setup import setup_demo_assets


def test_setup_demo_assets(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    demo_cfg = OmegaConf.create({})
    setup_demo_assets(demo_dir=demo_dir, demo_config=demo_cfg, requirements=["gradio"])
    assert (demo_dir / "app.py").exists()
    assert (demo_dir / "requirements.txt").exists()


def test_setup_demo_assets_writes_ui_templates(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    demo_cfg = OmegaConf.create({})
    setup_demo_assets(
        demo_dir=demo_dir,
        demo_config=demo_cfg,
        ui_templates={"basic": {"inputs": [], "outputs": []}},
    )
    assert (demo_dir / "ui_templates" / "basic.yaml").exists()
