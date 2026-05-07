"""Demo packaging helpers."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

from espnet3.utils.publish import (
    _copy_pack_include_paths,
)

logger = logging.getLogger(__name__)


def pack_demo(system) -> Path:
    """Package a demo into a self-contained directory ready for upload.

    Reads ``system.demo_config`` and produces a directory that contains:

    - ``demo.yaml``: the resolved demo config with all paths rewritten to be
      relative to the output directory so the pack is portable.
    - ``app.py``: the Gradio launcher script specified by
      ``demo_config.ui.app_script``.
    - Any extra files listed in ``demo_config.pack.include``.
    - Optionally a description file referenced by ``demo_config.ui.description``.

    The output directory is determined in order of priority:

    1. ``demo_config.pack.out_dir`` (explicit)
    2. ``system.exp_dir / "demo"``
    3. ``Path.cwd() / "demo"``

    Args:
        system: An ESPnet3 system instance with ``demo_config`` set.

    Returns:
        Path to the packed demo directory.

    Raises:
        RuntimeError: If ``system.demo_config`` is ``None``.
        ValueError: If ``demo_config.model.dir_or_tag`` cannot be inferred.

    Examples:
        >>> demo_dir = pack_demo(system)
        >>> print(demo_dir)
        /path/to/exp/demo
    """
    # --- validate ---
    demo_cfg = system.demo_config
    if demo_cfg is None:
        raise RuntimeError("pack_demo requires demo_config.")

    # --- resolve output directory ---
    pack_cfg = getattr(demo_cfg, "pack", None)
    demo_dir = _resolve_demo_out_dir(
        getattr(pack_cfg, "out_dir", None) if pack_cfg else None,
        system,
    )
    demo_dir.mkdir(parents=True, exist_ok=True)

    # --- write demo.yaml (paths rewritten to relative for portability) ---
    updated_cfg = _prepare_demo_config(demo_cfg, demo_dir, system)
    (demo_dir / "demo.yaml").write_text(
        OmegaConf.to_yaml(updated_cfg, resolve=True),
        encoding="utf-8",
    )

    # --- copy extra files declared in pack.include ---
    _copy_pack_includes(updated_cfg, demo_dir)

    # --- copy app.py and description asset ---
    _setup_demo_assets(demo_dir=demo_dir, demo_config=updated_cfg)
    return demo_dir


def upload_demo(system) -> None:
    """Upload a packed demo directory to a Hugging Face Space.

    Expects the demo to have been packed already via :func:`pack_demo`.
    Reads upload settings from ``system.demo_config.upload_demo``.

    Required config keys under ``demo_config.upload_demo``:

    - ``hf_repo``: full repository ID, e.g. ``"MyOrg/my-demo"``.

    Optional config keys:

    - ``repo_type``: HF repo type (default: ``"space"``).
    - ``organization``: HF organization name.
    - ``create``: extra options forwarded to the HF ``create_repo`` call.

    Args:
        system: An ESPnet3 system instance with ``demo_config`` set.

    Raises:
        RuntimeError: If ``demo_config``, ``upload_demo``, or ``hf_repo`` is
            missing, or if the packed demo directory does not exist.

    Examples:
        >>> upload_demo(system)
    """
    # --- validate config ---
    demo_cfg = system.demo_config
    if demo_cfg is None:
        raise RuntimeError("upload_demo requires demo_config.")
    upload_cfg = getattr(demo_cfg, "upload_demo", None)
    if upload_cfg is None:
        raise RuntimeError("upload_demo requires demo_config.upload_demo.")
    hf_repo_raw = getattr(upload_cfg, "hf_repo", None)
    if not hf_repo_raw:
        raise RuntimeError("upload_demo requires demo_config.upload_demo.hf_repo")

    # --- build repo / create options ---
    repo = str(hf_repo_raw)
    # HF API needs the repo name without the org prefix for create calls.
    create_repo_name = repo.rsplit("/", maxsplit=1)[-1]
    repo_type = getattr(upload_cfg, "repo_type", "space")
    create_cfg = getattr(upload_cfg, "create", None)
    create_opts = OmegaConf.to_container(create_cfg, resolve=True) if create_cfg else {}
    # Gradio and auto-confirm are safe defaults for a Spaces upload.
    create_opts.setdefault("space_sdk", "gradio")
    create_opts.setdefault("yes", True)
    if "organization" not in create_opts:
        organization = getattr(upload_cfg, "organization", None)
        if organization:
            create_opts["organization"] = organization

    # --- locate the packed demo directory ---
    pack_cfg = getattr(demo_cfg, "pack", None)
    demo_dir = _resolve_demo_out_dir(
        getattr(pack_cfg, "out_dir", None) if pack_cfg else None,
        system,
    )
    if not demo_dir.exists():
        raise RuntimeError(f"Demo pack not found: {demo_dir}")

    # --- upload ---
    from espnet3.utils.publish import _upload_common

    _upload_common(
        repo,
        demo_dir,
        repo_type=repo_type,
        create_options=create_opts,
        create_repo_name=create_repo_name,
    )


def _setup_demo_assets(demo_dir: Path, demo_config) -> None:
    """Copy the launcher and optional description asset into ``demo_dir``."""
    # --- copy app.py ---
    shutil.copy2(_resolve_app_script(demo_config), demo_dir / "app.py")

    # --- copy description file (optional) ---
    # _resolve_ui_description_path returns None when the file does not exist on
    # disk, so bail early rather than copying a missing file.
    description_src = _resolve_ui_description_path(demo_config)
    if description_src is None:
        return
    ui_cfg = getattr(demo_config, "ui", None)
    description_value = getattr(ui_cfg, "description", None) if ui_cfg else None
    if not description_value:
        return
    # Preserve the destination sub-path as written in the config so the demo
    # app can load it by the same name.
    dst = demo_dir / str(description_value)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(description_src, dst)


def _resolve_demo_out_dir(out_dir, system) -> Path:
    # explicit > exp_dir/demo > cwd/demo
    if out_dir:
        return Path(out_dir)
    exp_dir = getattr(system, "exp_dir", None)
    if exp_dir:
        return Path(exp_dir) / "demo"
    return Path.cwd() / "demo"


def _prepare_demo_config(demo_cfg, demo_dir: Path, system) -> DictConfig:
    # --- deep-copy to avoid mutating the live system config ---
    cfg = OmegaConf.create(OmegaConf.to_container(demo_cfg, resolve=True))

    # --- drop build-time keys that have no meaning at demo runtime ---
    # config_name is used by Hydra to select the config; the demo loader does
    # not understand it.
    pack_cfg = getattr(cfg, "pack", None)
    if pack_cfg is not None and "config_name" in pack_cfg:
        del pack_cfg["config_name"]

    # --- stamp the resolved output path so the demo app knows where it lives ---
    cfg.demo_dir = str(demo_dir)

    # --- ensure model section exists ---
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        cfg["model"] = OmegaConf.create({})
        model_cfg = cfg.model

    # --- infer dir_or_tag if not explicitly set ---
    if not model_cfg.get("dir_or_tag"):
        default_ref = None
        # Priority 1: explicit out_dir in publication_config.pack_model
        publication_cfg = getattr(system, "publication_config", None)
        if publication_cfg is not None:
            pack_cfg = getattr(publication_cfg, "pack_model", None)
            if pack_cfg is not None and getattr(pack_cfg, "out_dir", None):
                target = Path(pack_cfg.out_dir)
                target_path = (
                    target if target.is_absolute() else (Path.cwd() / target).resolve()
                )
                default_ref = os.path.relpath(target_path, start=demo_dir)
        # Priority 2: conventional location pack_model() uses when out_dir is
        # not set (see utils/publish.py line ~489).
        if default_ref is None:
            exp_dir = getattr(system, "exp_dir", None)
            if exp_dir is not None:
                target = Path(exp_dir) / "model_pack"
                target_path = (
                    target if target.is_absolute() else (Path.cwd() / target).resolve()
                )
                default_ref = os.path.relpath(target_path, start=demo_dir)
        if default_ref is not None:
            model_cfg["dir_or_tag"] = default_ref
    if not model_cfg.get("dir_or_tag"):
        raise ValueError(
            "demo_config.model.dir_or_tag is required when no local model pack "
            "can be inferred."
        )

    # --- normalize dir_or_tag to a relative path for portability ---
    # If it points to an existing local directory, store it relative to
    # demo_dir so the packed demo works on any machine.
    raw_ref = str(model_cfg["dir_or_tag"])
    candidate = Path(raw_ref).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    model_cfg["dir_or_tag"] = (
        os.path.relpath(candidate, start=demo_dir)
        if candidate.exists() and candidate.is_dir()
        else raw_ref
    )

    # --- normalize description path ---
    # If it was absolute (e.g. resolved from the repo root), strip to just the
    # filename so it matches the copied asset placed directly in demo_dir.
    ui_cfg = getattr(cfg, "ui", None)
    description_src = _resolve_ui_description_path(cfg)
    if ui_cfg is not None and description_src is not None:
        description_path = Path(str(ui_cfg.description))
        if description_path.is_absolute():
            ui_cfg.description = description_path.name
    return cfg


def _copy_pack_includes(demo_cfg, demo_dir: Path) -> None:
    # --- read include / exclude lists from config ---
    pack_cfg = getattr(demo_cfg, "pack", None)
    include_cfg = getattr(pack_cfg, "include", None) if pack_cfg else None
    if not include_cfg:
        return

    # normalise to flat str lists (config value may be a scalar or a ListConfig)
    raw_include_paths = (
        [str(path) for path in include_cfg]
        if isinstance(include_cfg, (list, tuple, ListConfig))
        else [str(include_cfg)]
    )

    # --- build exclude list ---
    # Always exclude the demo output dir itself and bytecode caches to avoid
    # recursive copies or bloating the pack with compiled files.
    exclude_patterns = [demo_dir.name, "__pycache__"]
    exclude_cfg = getattr(pack_cfg, "exclude", None) if pack_cfg else None
    if exclude_cfg:
        exclude_patterns += (
            [str(pattern) for pattern in exclude_cfg]
            if isinstance(exclude_cfg, (list, tuple, ListConfig))
            else [str(exclude_cfg)]
        )

    # --- copy ---
    _copy_pack_include_paths(
        include_paths=raw_include_paths,
        out_dir=demo_dir,
        recipe_root=Path.cwd(),
        exclude_patterns=exclude_patterns,
    )


def _resolve_app_script(demo_config) -> Path:
    # --- read from config ---
    ui_cfg = getattr(demo_config, "ui", None)
    explicit = getattr(ui_cfg, "app_script", None) if ui_cfg else None
    if not explicit:
        raise ValueError("demo_config.ui.app_script is required.")

    # --- resolve path (cwd-relative, then repo-root fallback) ---
    # Allow paths relative to the repo root (e.g. espnet3/publication/demo/app.py)
    # so built-in launchers can be referenced without an absolute path.
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
    # --- read from config ---
    ui_cfg = getattr(demo_config, "ui", None)
    description = getattr(ui_cfg, "description", None) if ui_cfg else None
    if not description:
        return None

    # --- resolve path (cwd-relative, then repo-root fallback) ---
    # Same repo-root fallback as _resolve_app_script: allows description files
    # to be referenced by their path relative to the repository.
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
