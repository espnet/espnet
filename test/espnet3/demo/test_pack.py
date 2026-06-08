from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from espnet3.publication.demo.packing import pack_demo
from espnet3.utils.config_utils import load_and_merge_config


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
    # Local model dir is symlinked into the bundle as model_pack.
    assert (demo_dir / "model_pack").is_symlink()
    demo_yaml = yaml.safe_load((demo_dir / "demo.yaml").read_text(encoding="utf-8"))
    assert demo_yaml["model"]["dir_or_tag"] == "model_pack"


def test_pack_demo_writes_requirements_txt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "packed_demo"
    model_pack_dir = tmp_path / "model_pack"
    model_pack_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    demo_cfg = OmegaConf.create(
        {
            "model": {"dir_or_tag": "espnet/some-model"},
            "ui": {"app_script": "egs3/TEMPLATE/asr/src/app.py"},
            "pack": {
                "out_dir": str(demo_dir),
                "requirements": [
                    "espnet",
                    "gradio",
                    "git+https://github.com/espnet/espnet@main",
                ],
            },
        }
    )

    class DummySystem:
        demo_config = demo_cfg
        exp_dir = None

    pack_demo(DummySystem())

    req = (demo_dir / "requirements.txt").read_text(encoding="utf-8")
    assert req == "espnet\ngradio\ngit+https://github.com/espnet/espnet@main\n"


def test_pack_demo_skips_requirements_txt_when_not_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "packed_demo"
    monkeypatch.chdir(tmp_path)
    demo_cfg = OmegaConf.create(
        {
            "model": {"dir_or_tag": "espnet/some-model"},
            "ui": {"app_script": "egs3/TEMPLATE/asr/src/app.py"},
            "pack": {"out_dir": str(demo_dir)},
        }
    )

    class DummySystem:
        demo_config = demo_cfg
        exp_dir = None

    pack_demo(DummySystem())

    assert not (demo_dir / "requirements.txt").exists()


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


def test_pack_demo_renders_space_readme(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "packed_demo"
    model_pack_dir = tmp_path / "model_pack"
    model_pack_dir.mkdir()
    readme_context = {
        "title": "Custom demo",
        "emoji": "microphone",
        "color_from": "indigo",
        "color_to": "purple",
        "sdk": "gradio",
        "python_version": "3.12",
        "app_file": "custom_app.py",
        "pinned": "true",
        "license": "mit",
        "tags": "- custom-tag\n- task-tag",
        "description": "Custom description.",
        "hf_repo": "custom/repo",
        "model_ref": "custom/model",
        "creator": "custom creator",
    }
    demo_config_path = tmp_path / "conf" / "demo.yaml"
    demo_config_path.parent.mkdir()
    app_script = tmp_path / "src" / "app.py"
    app_script.parent.mkdir()
    app_script.write_text("# test app\n", encoding="utf-8")
    demo_config = {
        "model": {
            "dir_or_tag": str(model_pack_dir),
        },
        "ui": {
            "title": "Ignored by readme_context",
        },
        "pack": {
            "out_dir": str(demo_dir),
            "readme_context": readme_context,
        },
        "upload_demo": {
            "hf_repo": "ignored/by-context",
        },
    }
    demo_config_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(demo_config)),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    demo_cfg = load_and_merge_config(
        demo_config_path,
        config_name="demo.yaml",
        default_package="egs3.TEMPLATE.asr",
        resolve=False,
    )

    class DummySystem:
        demo_config = demo_cfg
        exp_dir = None
        publish_config = None

    pack_demo(DummySystem())

    readme = (demo_dir / "README.md").read_text(encoding="utf-8")
    expected_lines = [
        "title: Custom demo",
        "emoji: microphone",
        "colorFrom: indigo",
        "colorTo: purple",
        "sdk: gradio",
        'python_version: "3.12"',
        "app_file: custom_app.py",
        "pinned: true",
        "license: mit",
        "tags:\n- custom-tag\n- task-tag",
        "# Custom demo",
        "Custom description.",
        "- Space: `custom/repo`",
        "- Model: `custom/model`",
        "- Creator: `custom creator`",
    ]
    for expected_line in expected_lines:
        assert expected_line in readme
