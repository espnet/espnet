"""Demo config loading helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from espnet3.utils.config_utils import load_config_with_defaults


def resolve_demo_config_path(
    demo_dir: str | Path,
    demo_config_path: str | Path | None = None,
) -> Path:
    """Resolve the packed demo config path.

    Args:
        demo_dir: Directory produced by ``pack_demo()``.
        demo_config_path: Optional explicit config path. Relative paths are
            resolved from ``demo_dir``.

    Returns:
        Absolute path to the selected demo config file.

    Raises:
        FileNotFoundError: If ``demo_dir`` does not exist, the explicit config
            path is missing, or no config candidate can be found.
        RuntimeError: If multiple config candidates are found.
    """
    demo_root = Path(demo_dir).resolve()
    if not demo_root.is_dir():
        raise FileNotFoundError(
            f"demo_dir must point to an existing directory: {demo_root}"
        )

    if demo_config_path is not None:
        config_path = Path(demo_config_path)
        if not config_path.is_absolute():
            config_path = demo_root / config_path
        config_path = config_path.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(
                f"demo config path does not exist: {config_path}"
            )
        return config_path

    candidates = []
    for pattern in ("*.yaml", "*.yml"):
        for path in sorted(demo_root.glob(pattern)):
            try:
                cfg = load_config_with_defaults(str(path))
            except Exception:
                continue
            model_cfg = getattr(cfg, "model", None)
            if model_cfg is not None and getattr(model_cfg, "dir_or_tag", None):
                candidates.append(path.resolve())

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Could not locate a demo config under: {demo_root}")
    raise RuntimeError(
        "Multiple demo config candidates found. Pass demo_config_path "
        f"explicitly. candidates={candidates}"
    )


def load_demo_config(
    demo_dir: str | Path,
    demo_config_path: str | Path | None = None,
) -> tuple[Path, DictConfig]:
    """Load a packed demo config.

    Args:
        demo_dir: Directory produced by ``pack_demo()``.
        demo_config_path: Optional explicit config path. Relative paths are
            resolved from ``demo_dir``.

    Returns:
        Tuple of ``(config_path, demo_config)``.
    """
    config_path = resolve_demo_config_path(demo_dir, demo_config_path)
    return config_path, load_config_with_defaults(str(config_path))
