"""Demo packaging helpers."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from espnet3.utils.publish import _render_readme

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
    _write_readme(system, demo_dir)
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
    """Copy the launcher and optional recipe UI module into ``demo_dir``."""
    launcher_path = demo_dir / str(demo_config.pack.launcher_name)
    if not launcher_path.exists():
        shutil.copy2(_resolve_app_script(demo_config), launcher_path)
    _copy_recipe_ui_module(demo_dir, demo_config)


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
        cfg.model = OmegaConf.create({})
        model_cfg = cfg.model
    if not getattr(model_cfg, "dir_or_tag", None):
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
            model_cfg.dir_or_tag = default_ref
    if not getattr(model_cfg, "dir_or_tag", None):
        raise ValueError(
            "demo_config.model.dir_or_tag is required when no local model pack "
            "can be inferred."
        )
    raw_ref = str(model_cfg.dir_or_tag)
    candidate = Path(raw_ref).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    model_cfg.dir_or_tag = (
        os.path.relpath(candidate, start=demo_dir)
        if candidate.exists() and candidate.is_dir()
        else raw_ref
    )
    return cfg


def _copy_pack_includes(demo_cfg, demo_dir: Path) -> None:
    pack_cfg = getattr(demo_cfg, "pack", None)
    include_paths = getattr(pack_cfg, "include", None) if pack_cfg else None
    if not include_paths:
        return
    for entry in include_paths:
        src_value = entry if isinstance(entry, str) else getattr(entry, "src", None)
        if src_value is None:
            raise ValueError("pack include entry has no resolvable src path.")
        src = Path(src_value)
        if not src.is_absolute():
            src = (Path.cwd() / src).resolve()
        try:
            dst = demo_dir / src.relative_to(Path.cwd())
        except ValueError:
            dst = demo_dir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(dst):
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _write_readme(system, demo_dir: Path) -> None:
    demo_cfg = system.demo_config
    pack_cfg = getattr(demo_cfg, "pack", None) if demo_cfg is not None else None
    readme_template_path = getattr(pack_cfg, "readme", None) if pack_cfg else None
    if not readme_template_path:
        return

    template_path = Path(readme_template_path)
    if not template_path.is_absolute() and not template_path.exists():
        repo_template_path = Path(__file__).resolve().parents[3] / template_path
        if repo_template_path.exists():
            template_path = repo_template_path
    if not template_path.exists():
        raise FileNotFoundError(f"README template not found: {template_path}")

    context = _build_readme_context(demo_cfg)
    context.update(dict(getattr(pack_cfg, "readme_context", {}) or {}))
    readme_text = _render_readme(template_path.read_text(encoding="utf-8"), context)
    (demo_dir / "README.md").write_text(readme_text, encoding="utf-8")


def _build_readme_context(demo_cfg) -> dict[str, str]:
    model_cfg = getattr(demo_cfg, "model", None) if demo_cfg is not None else None
    model_ref = str(getattr(model_cfg, "dir_or_tag", "")) if model_cfg else ""
    hf_upload = getattr(demo_cfg, "upload_demo", None) if demo_cfg is not None else None
    hf_repo = str(getattr(hf_upload, "hf_repo", "")) if hf_upload else ""
    ui_cfg = getattr(demo_cfg, "ui", None) if demo_cfg is not None else None
    title = str(getattr(ui_cfg, "title", "")) if ui_cfg else ""
    if hf_repo:
        usage_load_call = (
            f'model = InferenceModel.from_pretrained("{hf_repo}",'
            " trust_user_code=True)"
        )
    else:
        usage_load_call = (
            "model = InferenceModel.from_packed("
            '"path/to/demo", trust_user_code=True)'
        )

    description_val = getattr(ui_cfg, "description", None) if ui_cfg else None
    if not description_val:
        description = ""
    else:
        desc_path = Path(str(description_val))
        if desc_path.is_absolute():
            description = (
                desc_path.read_text(encoding="utf-8") if desc_path.is_file() else ""
            )
        else:
            candidate = (Path.cwd() / desc_path).resolve()
            if candidate.is_file():
                description = candidate.read_text(encoding="utf-8")
            elif desc_path.suffix.lower() in {".md", ".txt"}:
                description = ""
            else:
                description = str(description_val)

    return {
        "title": title or "ESPnet3 Demo",
        "description": description,
        "model_ref": model_ref,
        "recipe": "",
        "creator": "",
        "usage_load_call": usage_load_call,
    }


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


def _copy_recipe_ui_module(demo_dir: Path, demo_config) -> None:
    ui_cfg = getattr(demo_config, "ui", None)
    registry_path = getattr(ui_cfg, "asset_registry", None) if ui_cfg else None
    if not registry_path:
        return
    src_path = Path(registry_path)
    if not src_path.is_absolute():
        src_path = (Path.cwd() / src_path).resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"demo_config.ui.asset_registry not found: {src_path}")
    bundle_path = Path(str(registry_path))
    bundle_rel = Path(bundle_path.name) if bundle_path.is_absolute() else bundle_path
    dst_path = demo_dir / bundle_rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    init_py = src_path.parent / "__init__.py"
    if init_py.is_file():
        shutil.copy2(init_py, dst_path.parent / "__init__.py")
