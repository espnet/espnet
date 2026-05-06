"""Demo packaging helpers."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

from espnet3.utils.publish import (
    _build_pack_ignore,
    _copy_path,
    _expand_pack_paths,
)

logger = logging.getLogger(__name__)


def pack_demo(system) -> Path:
    """Package demo assets and configs into a demo directory."""
    demo_cfg = system.demo_config
    if demo_cfg is None:
        raise RuntimeError("pack_demo requires demo_config.")
    pack_cfg = getattr(demo_cfg, "pack", None)
    demo_dir = _resolve_demo_out_dir(
        getattr(pack_cfg, "out_dir", None) if pack_cfg else None,
        system,
    )
    demo_dir.mkdir(parents=True, exist_ok=True)

    updated_cfg = _prepare_demo_config(demo_cfg, demo_dir, system)
    (demo_dir / "demo.yaml").write_text(
        OmegaConf.to_yaml(updated_cfg, resolve=True),
        encoding="utf-8",
    )
    _copy_pack_includes(updated_cfg, demo_dir)
    setup_demo_assets(demo_dir=demo_dir, demo_config=updated_cfg)
    return demo_dir


def upload_demo(system) -> None:
    """Upload demo assets to a remote destination (not implemented)."""
    demo_cfg = system.demo_config
    if demo_cfg is None:
        raise RuntimeError("upload_demo requires demo_config.")
    upload_cfg = getattr(demo_cfg, "upload_demo", None)
    if upload_cfg is None:
        raise RuntimeError("upload_demo requires demo_config.upload_demo.")
    hf_repo_raw = getattr(upload_cfg, "hf_repo", None)
    if not hf_repo_raw:
        raise RuntimeError("upload_demo requires demo_config.upload_demo.hf_repo")
    repo = str(hf_repo_raw)
    create_repo_name = repo.rsplit("/", maxsplit=1)[-1]
    repo_type = getattr(upload_cfg, "repo_type", "space")
    create_cfg = getattr(upload_cfg, "create", None)
    create_opts = OmegaConf.to_container(create_cfg, resolve=True) if create_cfg else {}
    if "organization" not in create_opts:
        organization = getattr(upload_cfg, "organization", None)
        if organization:
            create_opts["organization"] = organization
    if "space_sdk" not in create_opts:
        space_sdk = getattr(upload_cfg, "space_sdk", None)
        if space_sdk:
            create_opts["space_sdk"] = space_sdk
    if "yes" not in create_opts and hasattr(upload_cfg, "yes"):
        create_opts["yes"] = bool(getattr(upload_cfg, "yes"))

    pack_cfg = getattr(demo_cfg, "pack", None)
    demo_dir = _resolve_demo_out_dir(
        getattr(pack_cfg, "out_dir", None) if pack_cfg else None,
        system,
    )
    if not demo_dir.exists():
        raise RuntimeError(f"Demo pack not found: {demo_dir}")

    from espnet3.utils.publish import _upload_common

    _upload_common(
        repo,
        demo_dir,
        repo_type=repo_type,
        create_options=create_opts,
        create_repo_name=create_repo_name,
    )


def setup_demo_assets(demo_dir: Path, demo_config) -> None:
    """Copy the launcher and optional description asset into ``demo_dir``."""
    shutil.copy2(_resolve_app_script(demo_config), demo_dir / "app.py")
    description_src = _resolve_ui_description_path(demo_config)
    if description_src is None:
        return
    ui_cfg = getattr(demo_config, "ui", None)
    description_value = getattr(ui_cfg, "description", None) if ui_cfg else None
    if not description_value:
        return
    dst = demo_dir / str(description_value)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(description_src, dst)


def _resolve_demo_out_dir(out_dir, system) -> Path:
    if out_dir:
        return Path(out_dir)
    exp_dir = getattr(system, "exp_dir", None)
    if exp_dir:
        return Path(exp_dir) / "demo"
    return Path.cwd() / "demo"


def _prepare_demo_config(demo_cfg, demo_dir: Path, system) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(demo_cfg, resolve=True))
    pack_cfg = getattr(cfg, "pack", None)
    if pack_cfg is not None and "config_name" in pack_cfg:
        del pack_cfg["config_name"]
    cfg.demo_dir = str(demo_dir)
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        cfg["model"] = OmegaConf.create({})
        model_cfg = cfg.model
    if not model_cfg.get("dir-or-tag"):
        default_ref = None
        publication_cfg = getattr(system, "publication_config", None)
        if publication_cfg is not None:
            pack_cfg = getattr(publication_cfg, "pack_model", None)
            if pack_cfg is not None and getattr(pack_cfg, "out_dir", None):
                target = Path(pack_cfg.out_dir)
                target_path = (
                    target if target.is_absolute() else (Path.cwd() / target).resolve()
                )
                default_ref = os.path.relpath(target_path, start=demo_dir)
        if default_ref is None:
            exp_dir = getattr(system, "exp_dir", None)
            if exp_dir is not None:
                target = Path(exp_dir) / "model_pack"
                target_path = (
                    target if target.is_absolute() else (Path.cwd() / target).resolve()
                )
                default_ref = os.path.relpath(target_path, start=demo_dir)
        if default_ref is not None:
            model_cfg["dir-or-tag"] = default_ref
    if not model_cfg.get("dir-or-tag"):
        raise ValueError(
            "demo_config.model.dir-or-tag is required when no local model pack "
            "can be inferred."
        )
    raw_ref = str(model_cfg["dir-or-tag"])
    candidate = Path(raw_ref).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    model_cfg["dir-or-tag"] = (
        os.path.relpath(candidate, start=demo_dir)
        if candidate.exists() and candidate.is_dir()
        else raw_ref
    )
    ui_cfg = getattr(cfg, "ui", None)
    description_src = _resolve_ui_description_path(cfg)
    if ui_cfg is not None and description_src is not None:
        description_path = Path(str(ui_cfg.description))
        if description_path.is_absolute():
            ui_cfg.description = description_path.name
    return cfg


def _copy_pack_includes(demo_cfg, demo_dir: Path) -> None:
    pack_cfg = getattr(demo_cfg, "pack", None)
    include_cfg = getattr(pack_cfg, "include", None) if pack_cfg else None
    if not include_cfg:
        return

    raw_include_paths = (
        [str(path) for path in include_cfg]
        if isinstance(include_cfg, (list, tuple, ListConfig))
        else [str(include_cfg)]
    )
    exclude_patterns = [demo_dir.name, "__pycache__"]
    exclude_cfg = getattr(pack_cfg, "exclude", None) if pack_cfg else None
    if exclude_cfg:
        exclude_patterns += (
            [str(pattern) for pattern in exclude_cfg]
            if isinstance(exclude_cfg, (list, tuple, ListConfig))
            else [str(exclude_cfg)]
        )

    include_paths = _expand_pack_paths(raw_include_paths, Path.cwd())
    for src in include_paths:
        if not src.exists():
            logger.warning("Pack include path does not exist: %s", src)
            continue
        src = src.resolve()
        try:
            dst = demo_dir / src.relative_to(Path.cwd())
        except ValueError:
            dst = demo_dir / src.name
        if os.path.lexists(dst):
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        ignore = _build_pack_ignore(src, exclude_patterns) if src.is_dir() else None
        _copy_path(src=src, dst=dst, ignore=ignore)
def _resolve_app_script(demo_config) -> Path:
    ui_cfg = getattr(demo_config, "ui", None)
    explicit = getattr(ui_cfg, "app_script", None) if ui_cfg else None
    if not explicit:
        raise ValueError("demo_config.ui.app_script is required.")
    path = Path(explicit)
    if not path.is_absolute() and not path.exists():
        repo_path = Path(__file__).resolve().parents[3] / path
        if repo_path.exists():
            path = repo_path
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"demo_config.ui.app_script not found: {path}")
    return path


def _resolve_ui_description_path(demo_config) -> Path | None:
    ui_cfg = getattr(demo_config, "ui", None)
    description = getattr(ui_cfg, "description", None) if ui_cfg else None
    if not description:
        return None
    path = Path(str(description))
    if not path.is_absolute() and not path.exists():
        repo_path = Path(__file__).resolve().parents[3] / path
        if repo_path.exists():
            path = repo_path
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        return None
    return path
