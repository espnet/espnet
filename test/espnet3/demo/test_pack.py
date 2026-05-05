from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from espnet3.publication.demo.packer import pack_demo


def test_pack_demo_writes_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "packed_demo"
    model_pack_dir = tmp_path / "model_pack"
    model_pack_dir.mkdir()
    extra_src = tmp_path / "extra.txt"
    extra_src.write_text("hello\n", encoding="utf-8")
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "ui_assets.py").write_text("# recipe ui\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    demo_cfg = OmegaConf.create(
        {
            "model": {
                "dir_or_tag": str(model_pack_dir),
            },
            "ui": {
                "asset_registry": "custom/ui_assets.py",
            },
            "pack": {
                "out_dir": str(demo_dir),
                "config_name": "packed_demo.yaml",
                "launcher_name": "app.py",
                "readme": "egs3/TEMPLATE/asr/src/demo_readme_template.md",
                "include": [str(extra_src)],
            },
        }
    )

    class DummySystem:
        demo_config = demo_cfg
        exp_dir = None
        publish_config = None

    out_dir = pack_demo(DummySystem())
    assert out_dir == demo_dir
    assert (demo_dir / "packed_demo.yaml").exists()
    assert (demo_dir / extra_src.name).read_text(encoding="utf-8") == "hello\n"
    assert (demo_dir / "app.py").exists()
    assert (demo_dir / "README.md").exists()
    assert (demo_dir / "custom" / "ui_assets.py").exists()
    demo_yaml = yaml.safe_load(
        (demo_dir / "packed_demo.yaml").read_text(encoding="utf-8")
    )
    assert demo_yaml["model"]["dir_or_tag"] == "../model_pack"


def test_pack_demo_skips_readme_when_template_not_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "packed_demo"
    model_pack_dir = tmp_path / "model_pack"
    model_pack_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    demo_cfg = OmegaConf.create(
        {
            "model": {
                "dir_or_tag": str(model_pack_dir),
            },
            "pack": {
                "out_dir": str(demo_dir),
                "config_name": "packed_demo.yaml",
                "launcher_name": "app.py",
                "include": [],
            },
        }
    )

    class DummySystem:
        demo_config = demo_cfg
        exp_dir = None
        publish_config = None

    out_dir = pack_demo(DummySystem())

    assert out_dir == demo_dir
    assert not (demo_dir / "README.md").exists()
