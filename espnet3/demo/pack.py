"""Demo packaging helpers."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from espnet3.demo.resolve import (
    load_infer_config,
    resolve_absolute_path,
    resolve_infer_path,
    resolve_provider_class,
)
from espnet3.demo.setup import setup_demo_assets
from espnet3.utils.publish import _load_readme_template, _write_readme

logger = logging.getLogger(__name__)


def pack_demo(system) -> Path:
    demo_cfg = system.demo_config
    if demo_cfg is None:
        raise RuntimeError("pack_demo requires demo_config.")
    demo_cfg_path = system.demo_config_path
    pack_cfg = getattr(demo_cfg, "pack", None)
    if resolve_provider_class(demo_cfg) is None:
        raise ValueError("inference provider is not configured for this system.")
    demo_dir = _resolve_demo_out_dir(
        getattr(pack_cfg, "out_dir", None) if pack_cfg else None,
        system,
    )
    demo_dir.mkdir(parents=True, exist_ok=True)

    infer_path = resolve_infer_path(
        getattr(demo_cfg, "infer_config", None),
        demo_cfg_path,
    )
    if infer_path is None:
        raise ValueError("infer_config is required for demo setup.")
    infer_cfg = load_infer_config(infer_path)
    updated_cfg = _prepare_demo_config(demo_cfg, demo_dir, infer_path)
    _write_demo_config(demo_dir, updated_cfg)

    _write_infer_config(demo_dir, infer_cfg)

    _copy_pack_files(demo_cfg, demo_dir)
    setup_demo_assets(demo_dir=demo_dir, demo_config=updated_cfg)
    _write_readme_if_missing(system, demo_dir, infer_cfg)
    return demo_dir


def upload_demo(system) -> None:
    raise RuntimeError("upload_demo is not implemented yet.")


def _prepare_demo_config(demo_cfg, demo_dir: Path, infer_path: Path | None) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(demo_cfg, resolve=True))
    cfg.demo_dir = str(demo_dir)
    if infer_path is not None:
        cfg.infer_config = "config/infer.yaml"
    ui_cfg = getattr(cfg, "ui", None)
    if ui_cfg is not None and not getattr(ui_cfg, "article_path", None):
        ui_cfg.article_path = "README.md"
    return cfg


def _write_demo_config(demo_dir: Path, demo_cfg: DictConfig) -> None:
    (demo_dir / "demo.yaml").write_text(
        OmegaConf.to_yaml(demo_cfg, resolve=True),
        encoding="utf-8",
    )


def _resolve_demo_out_dir(out_dir, system) -> Path:
    if out_dir:
        return Path(out_dir)
    exp_dir = getattr(system, "exp_dir", None)
    if exp_dir:
        return Path(exp_dir) / "demo"
    return Path.cwd() / "demo"


def _resolve_dest_path(path_value, src: Path, demo_dir: Path) -> Path:
    if path_value:
        candidate = Path(path_value)
        if not candidate.is_absolute():
            return demo_dir / candidate
        try:
            return demo_dir / src.resolve().relative_to(Path.cwd())
        except Exception:
            return demo_dir / src.name
    return demo_dir / src.name


def _copy_pack_files(demo_cfg, demo_dir: Path) -> None:
    pack_cfg = getattr(demo_cfg, "pack", None)
    files = getattr(pack_cfg, "files", None) if pack_cfg else None
    if not files:
        return
    for entry in list(files):
        src_value = entry if isinstance(entry, str) else getattr(entry, "src", None)
        src = resolve_absolute_path(src_value, base=Path.cwd())
        dst_path = _resolve_dest_path(src_value, src, demo_dir)
        _copy_path(src, dst_path, symlink=True)


def _copy_path(src: Path, dst: Path, *, symlink: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst):
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if symlink:
        os.symlink(src, dst)
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _write_infer_config(demo_dir: Path, infer_cfg: DictConfig) -> None:
    out_cfg = OmegaConf.create(OmegaConf.to_container(infer_cfg, resolve=True))
    out_path = demo_dir / "config" / "infer.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(OmegaConf.to_yaml(out_cfg, resolve=True), encoding="utf-8")


def _write_readme_if_missing(system, demo_dir: Path, infer_cfg) -> None:
    readme_path = demo_dir / "README.md"
    if readme_path.exists():
        return
    pack_cfg = getattr(getattr(system, "publish_config", None), "pack_model", None)
    pack_cfg = pack_cfg or OmegaConf.create({})
    template = _load_readme_template(pack_cfg)
    template = _strip_top_tag_block(template)
    exp_dir = None
    if infer_cfg is not None:
        exp_dir_value = getattr(infer_cfg, "exp_dir", None)
        if exp_dir_value:
            exp_dir = Path(exp_dir_value)
    _write_readme(
        readme_template=template,
        out_dir=demo_dir,
        publish_cfg=OmegaConf.create({}),
        pack_cfg=pack_cfg,
        exp_dir=exp_dir,
        strategy="demo",
        system=system,
        scores_path=None,
        minimal=True,
    )


def _strip_top_tag_block(template_text: str) -> str:
    lines = template_text.splitlines()
    if len(lines) >= 3 and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :]).lstrip()
    return template_text

