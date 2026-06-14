"""Demo packaging helpers."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from omegaconf import DictConfig, ListConfig, OmegaConf

from espnet3.utils.publication_utils import (
    _build_pack_ignore,
    _copy_path,
    _expand_pack_paths,
    _infer_creator,
    _render_readme,
    _resolve_readme_template_path,
)

logger = logging.getLogger(__name__)


def pack_demo(system) -> Path:
    """Package a demo into a self-contained directory ready for upload.

    Reads ``system.demo_config`` and produces a directory that contains:

    - ``demo.yaml``: the resolved demo config with all paths rewritten to be
      relative to the output directory so the pack is portable.
    - ``app.py``: the Gradio launcher script specified by
      ``demo_config.ui.app_script``.
    - ``README.md``: optional Hugging Face Space card rendered from
      ``demo_config.pack.readme``.
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
    _link_local_model_into_bundle(updated_cfg, demo_dir)
    (demo_dir / "demo.yaml").write_text(
        OmegaConf.to_yaml(updated_cfg, resolve=True),
        encoding="utf-8",
    )

    # --- write requirements.txt when pack.requirements is set ---
    _write_requirements(updated_cfg, demo_dir)

    # --- copy extra files declared in pack.include ---
    _copy_pack_includes(updated_cfg, demo_dir)

    # --- copy app.py and description asset ---
    _setup_demo_assets(demo_dir=demo_dir, demo_config=updated_cfg)

    # --- write README.md for Hugging Face Spaces when configured ---
    _write_demo_readme(updated_cfg, demo_dir, system)
    return demo_dir


def upload_demo(system) -> None:
    """Upload a packed demo directory to a Hugging Face Space.

    Expects the demo to have been packed already via :func:`pack_demo`.
    Reads upload settings from ``system.demo_config.upload_demo``.

    Required config keys under ``demo_config.upload_demo``:

    - ``hf_repo``: full repository ID, e.g. ``"MyOrg/my-demo"``.

    Optional config keys:

    - ``repo_type``: HF repo type (default: ``"space"``).
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
    organization = getattr(upload_cfg, "organization", None)
    if organization and "/" not in repo:
        repo = f"{organization}/{repo}"
    repo_type = getattr(upload_cfg, "repo_type", "space")
    create_cfg = getattr(upload_cfg, "create", None)
    create_opts = OmegaConf.to_container(create_cfg, resolve=True) if create_cfg else {}
    create_opts.pop("yes", None)
    create_opts.setdefault("space_sdk", "gradio")

    # --- locate the packed demo directory ---
    pack_cfg = getattr(demo_cfg, "pack", None)
    demo_dir = _resolve_demo_out_dir(
        getattr(pack_cfg, "out_dir", None) if pack_cfg else None,
        system,
    )
    if not demo_dir.exists():
        raise RuntimeError(f"Demo pack not found: {demo_dir}")

    # --- upload ---
    logger.info("Ensuring Hugging Face demo repo exists: %s", repo)
    api = HfApi()
    private = bool(getattr(upload_cfg, "private", False))
    try:
        api.create_repo(
            repo_id=repo,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
            **create_opts,
        )
    except (HfHubHTTPError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to create Hugging Face demo repo '{repo}': {exc}"
        ) from exc
    logger.info("Uploading %s -> %s", demo_dir, repo)
    try:
        api.upload_folder(
            repo_id=repo,
            repo_type=repo_type,
            folder_path=str(demo_dir),
        )
    except (HfHubHTTPError, ValueError) as exc:
        raise RuntimeError(f"Failed to upload demo pack to '{repo}': {exc}") from exc
    logger.info("Demo upload complete: https://huggingface.co/spaces/%s", repo)


def _write_requirements(demo_cfg, demo_dir: Path) -> None:
    """Write ``requirements.txt`` from ``pack.requirements`` into the demo bundle.

    Entries may be any pip-compatible specifier, including ``git+https://``
    URLs. When ``pack.requirements`` is absent or empty, no file is written.

    Args:
        demo_cfg: Resolved demo config with an optional ``pack.requirements``
            list.
        demo_dir: Packed demo output directory where ``requirements.txt`` is
            written.

    Examples:
        Config with a mix of PyPI and git installs:

        .. code-block:: yaml

            pack:
              requirements:
                - espnet
                - gradio
                - git+https://github.com/espnet/espnet@main

        Produces ``demo_dir/requirements.txt``::

            espnet
            gradio
            git+https://github.com/espnet/espnet@main
    """
    pack_cfg = getattr(demo_cfg, "pack", None)
    requirements = getattr(pack_cfg, "requirements", None) if pack_cfg else None
    if not requirements:
        return
    lines = [str(r) for r in requirements]
    dest = demo_dir / "requirements.txt"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote requirements.txt | %d packages", len(lines))


def _link_local_model_into_bundle(demo_cfg, demo_dir: Path) -> None:
    """Symlink a local model directory into the demo bundle as ``model_pack``.

    Called by :func:`pack_demo` after :func:`_prepare_demo_config` has
    normalised ``model.dir_or_tag`` to a path relative to ``demo_dir``.
    When that relative path resolves to a directory that lives *outside*
    ``demo_dir``, a symlink ``demo_dir/model_pack -> <candidate>`` is created
    so the bundle references the model without copying it. :func:`upload_demo`
    resolves symlinks before uploading so HF receives the actual files.

    If ``dir_or_tag`` is already inside ``demo_dir`` or points to a Hugging
    Face tag (non-existent local path), this function is a no-op.

    Args:
        demo_cfg: Resolved demo config returned by :func:`_prepare_demo_config`.
            Mutated in place when a symlink is created.
        demo_dir: Packed demo output directory.
    """
    model_cfg = getattr(demo_cfg, "model", None)
    if model_cfg is None:
        return
    dir_or_tag = model_cfg.get("dir_or_tag") if hasattr(model_cfg, "get") else None
    if not dir_or_tag:
        return

    candidate = Path(str(dir_or_tag))
    if not candidate.is_absolute():
        candidate = (demo_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.is_dir():
        return

    # Already inside demo_dir — nothing to do.
    try:
        candidate.relative_to(demo_dir.resolve())
        return
    except ValueError:
        pass

    dest = demo_dir / "model_pack"
    logger.info(
        "Symlinking local model dir into demo bundle | src=%s dest=%s",
        candidate,
        dest,
    )
    dest.symlink_to(candidate, target_is_directory=True)
    model_cfg["dir_or_tag"] = "model_pack"


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

    # --- validate model.dir_or_tag ---
    # Set model.dir_or_tag in demo.yaml, or run pack_demo together with
    # publication_config so apply_training_experiment_context can propagate
    # pack_model.out_dir automatically.
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None or not model_cfg.get("dir_or_tag"):
        raise ValueError(
            "demo_config.model.dir_or_tag is required. "
            "Set it in demo.yaml or run pack_demo with publication_config "
            "so pack_model.out_dir is propagated automatically."
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
    recipe_root = Path.cwd().resolve()
    for path in _expand_pack_paths(raw_include_paths, recipe_root):
        if not path.exists():
            logger.warning("Demo include path does not exist: %s", path)
            continue
        try:
            dest = demo_dir / path.absolute().relative_to(recipe_root)
        except ValueError:
            logger.warning(
                "Demo include path is outside recipe root and will be placed "
                "at the bundle root as '%s': %s",
                path.name,
                path,
            )
            dest = demo_dir / path.name
        ignore = _build_pack_ignore(path, exclude_patterns) if path.is_dir() else None
        _copy_path(src=path, dst=dest, ignore=ignore)


def _write_demo_readme(demo_cfg, demo_dir: Path, system) -> None:
    """Render the optional Hugging Face Space README."""
    pack_cfg = getattr(demo_cfg, "pack", None)
    readme_template_path = getattr(pack_cfg, "readme", None) if pack_cfg else None
    if not readme_template_path:
        return
    template_path = _resolve_readme_template_path(str(readme_template_path))
    context = _build_demo_readme_context(demo_cfg)
    context.update(dict(getattr(pack_cfg, "readme_context", {}) or {}))
    readme_text = _render_readme(template_path.read_text(encoding="utf-8"), context)
    (demo_dir / "README.md").write_text(readme_text, encoding="utf-8")


def _build_demo_readme_context(demo_cfg) -> dict[str, str]:
    """Build runtime-only template values for a Hugging Face Space README.

    Returns only values that cannot be expressed in static config: installed
    package versions, Python version, creator from env, and the description
    string (which requires resolving whether ``ui.description`` is a file path).

    All other README variables (title, hf_repo, model_ref, emoji, sdk, license,
    tags, app_file, pinned, …) are defined under ``pack.readme_context`` in
    ``TEMPLATE/asr/conf/demo.yaml`` and are already merged into ``demo_cfg``
    by ``load_and_merge_config``. They are applied in ``_write_demo_readme``
    via ``context.update(pack_cfg.readme_context)``.
    """
    ui_cfg = getattr(demo_cfg, "ui", None)
    description = getattr(ui_cfg, "description", None) if ui_cfg else None
    if description is not None and _resolve_ui_description_path(demo_cfg) is None:
        description_text = str(description)
    else:
        description_text = ""

    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "description": description_text,
        "creator": _infer_creator(),
    }


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
