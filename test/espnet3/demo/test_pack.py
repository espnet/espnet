from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from espnet3.publication.demo.packing import pack_demo


def test_pack_demo_writes_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "packed_demo"
    model_pack_dir = tmp_path / "model_pack"
    model_pack_dir.mkdir()
    extra_src = tmp_path / "extra.txt"
    extra_src.write_text("hello\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    demo_cfg = OmegaConf.create(
        {
            "model": {
                "dir_or_tag": str(model_pack_dir),
            },
            "ui": {
                "app_script": "egs3/TEMPLATE/asr/src/app.py",
                "description": "README.md",
            },
            "pack": {
                "out_dir": str(demo_dir),
                "include": [str(extra_src)],
            },
        }
    )
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")

    class DummySystem:
        demo_config = demo_cfg
        exp_dir = None
        publish_config = None

    out_dir = pack_demo(DummySystem())
    assert out_dir == demo_dir
    assert (demo_dir / "demo.yaml").exists()
    assert (demo_dir / extra_src.name).read_text(encoding="utf-8") == "hello\n"
    assert (demo_dir / "app.py").exists()
    assert (demo_dir / "README.md").read_text(encoding="utf-8") == "# Demo\n"
    demo_yaml = yaml.safe_load((demo_dir / "demo.yaml").read_text(encoding="utf-8"))
    assert demo_yaml["model"]["dir_or_tag"] == "../model_pack"


def test_pack_demo_skips_description_copy_when_ui_description_is_missing(
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
            "ui": {
                "app_script": "egs3/TEMPLATE/asr/src/app.py",
            },
            "pack": {
                "out_dir": str(demo_dir),
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


def test_pack_demo_supports_exclude_patterns_for_included_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "packed_demo"
    model_pack_dir = tmp_path / "model_pack"
    model_pack_dir.mkdir()
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    (asset_dir / "keep.txt").write_text("keep\n", encoding="utf-8")
    (asset_dir / "debug.log").write_text("skip\n", encoding="utf-8")
    (asset_dir / "__pycache__").mkdir()
    (asset_dir / "__pycache__" / "cache.pyc").write_bytes(b"x")
    nested_dir = asset_dir / "tensorboard"
    nested_dir.mkdir()
    (nested_dir / "events.out").write_text("skip\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    demo_cfg = OmegaConf.create(
        {
            "model": {
                "dir_or_tag": str(model_pack_dir),
            },
            "ui": {
                "app_script": "egs3/TEMPLATE/asr/src/app.py",
            },
            "pack": {
                "out_dir": str(demo_dir),
                "include": [str(asset_dir)],
                "exclude": ["**/*.log", "**/tensorboard/**"],
            },
        }
    )

    class DummySystem:
        demo_config = demo_cfg
        exp_dir = None
        publish_config = None

    pack_demo(DummySystem())

    assert (demo_dir / "assets" / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert not (demo_dir / "assets" / "debug.log").exists()
    assert not (demo_dir / "assets" / "__pycache__").exists()
    assert not (demo_dir / "assets" / "tensorboard").exists()


def test_pack_demo_expands_globbed_include_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "packed_demo"
    model_pack_dir = tmp_path / "model_pack"
    model_pack_dir.mkdir()
    manifest_dir = tmp_path / "data" / "manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "train.tsv").write_text("utt\tpath\n", encoding="utf-8")
    (manifest_dir / "dev.tsv").write_text("utt\tpath\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    demo_cfg = OmegaConf.create(
        {
            "model": {
                "dir_or_tag": str(model_pack_dir),
            },
            "ui": {
                "app_script": "egs3/TEMPLATE/asr/src/app.py",
            },
            "pack": {
                "out_dir": str(demo_dir),
                "include": ["data/manifest/*.tsv"],
            },
        }
    )

    class DummySystem:
        demo_config = demo_cfg
        exp_dir = None
        publish_config = None

    pack_demo(DummySystem())

    assert (demo_dir / "data" / "manifest" / "train.tsv").read_text(
        encoding="utf-8"
    ) == "utt\tpath\n"
    assert (demo_dir / "data" / "manifest" / "dev.tsv").read_text(
        encoding="utf-8"
    ) == "utt\tpath\n"
