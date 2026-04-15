"""Publication helpers for ESPnet3."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

import espnet2
from espnet2.main_funcs.pack_funcs import pack as espnet2_pack

logger = logging.getLogger(__name__)
_RECIPE_BUNDLE_ENTRIES = (
    "conf",
    "src",
    "run.py",
    "pixi.toml",
    "pixi.lock",
    ".python-version",
)
_PATH_LIKE_KEYS = {
    "asr_train_config",
    "bpemodel",
    "data_dir",
    "exp_dir",
    "inference_dir",
    "lm_train_config",
    "manifest_path",
    "save_path",
    "token_list",
    "word_lm_train_config",
}
_YAML_SUFFIXES = (".yaml", ".yml")
_LOAD_YAML_PATTERN = re.compile(r"\$\{load_yaml:(?P<path>[^,}]+)")


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    """Run a subprocess command and return stdout."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _strip_optional_quotes(raw_value: str) -> str:
    """Return ``raw_value`` without matching surrounding quotes."""
    if (
        len(raw_value) >= 2
        and raw_value[0] == raw_value[-1]
        and raw_value[0] in ("'", '"')
    ):
        return raw_value[1:-1].strip()
    return raw_value


def _run_allow_repo_exists(cmd: List[str], cwd: Optional[Path] = None) -> str:
    """Run a subprocess command, allowing repo-exists errors."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}".lower()
        if "already exists" in combined or "409" in combined or "conflict" in combined:
            logger.info("Repo already exists; skipping create.")
            return result.stdout.strip()
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _repo_exists(repo: str, repo_type: str) -> Optional[bool]:
    """Check if a Hugging Face repo exists; returns None when unavailable."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except Exception:
        logger.warning("huggingface_hub not available; skipping repo check.")
        return None
    api = HfApi()
    try:
        api.repo_info(repo_id=repo, repo_type=repo_type)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as exc:
        logger.warning("Failed to check repo existence: %s", exc)
        return None


def _create_repo(
    repo: str,
    *,
    repo_type: str,
    organization: Optional[str] = None,
    space_sdk: Optional[str] = None,
    yes: bool = True,
) -> None:
    cmd = ["huggingface-cli", "repo", "create", repo, "--type", repo_type]
    if organization:
        cmd += ["--organization", organization]
    if repo_type == "space" and space_sdk:
        cmd += ["--space_sdk", space_sdk]
    if yes:
        cmd += ["-y"]
    _run_allow_repo_exists(cmd)


def _resolve_espnet2_spec(pack_cfg: DictConfig) -> dict:
    """Return espnet2 pack spec from config."""
    if isinstance(pack_cfg, DictConfig):
        pack_cfg = OmegaConf.to_container(pack_cfg, resolve=True) or {}
    if not isinstance(pack_cfg, dict):
        raise RuntimeError("pack_model.pack_cfg must be a mapping.")
    spec = pack_cfg.get("espnet2")
    if spec is None:
        raise RuntimeError("pack_model.strategy=espnet2 requires pack_model.espnet2.")
    if not isinstance(spec, dict):
        raise RuntimeError("pack_model.espnet2 must be a mapping.")
    if not spec.get("task"):
        raise RuntimeError(
            "pack_model.strategy=espnet2 requires pack_model.espnet2.task."
        )
    files_cfg = spec.get("files") or {}
    yaml_files_cfg = spec.get("yaml_files") or {}
    option_cfg = spec.get("option") or []
    return {
        "files": dict(files_cfg),
        "yaml_files": dict(yaml_files_cfg),
        "option": list(option_cfg),
    }


def _load_readme_template(pack_cfg: DictConfig) -> str:
    """Return README template content."""
    template = getattr(pack_cfg, "readme_template", None)
    if template is not None:
        template_path = Path(template)
        if not template_path.is_absolute():
            template_path = Path(__file__).resolve().parents[2] / template_path
        return template_path.read_text(encoding="utf-8")
    from egs3.TEMPLATE.asr.src.get_readme import get_readme

    return get_readme()


def _write_readme(
    *,
    readme_template: str,
    out_dir: Path,
    publication_cfg: DictConfig,
    pack_cfg: DictConfig,
    exp_dir: Path | None,
    strategy: str,
    system,
    scores_path: Optional[Path],
    minimal: bool = False,
) -> None:
    """Render README template into output directory if present."""
    if not readme_template:
        return
    git_info = _git_info()
    context = {
        "hf_repo": getattr(
            getattr(publication_cfg, "upload_model", None), "hf_repo", ""
        ),
        "system": system.__class__.__name__,
        "recipe": _resolve_recipe(system),
        "creator": _hf_username(),
        "pack_name": out_dir.name,
        "pack_strategy": strategy,
        "exp_dir": str(exp_dir) if exp_dir else "",
        "created_at": "" if minimal else datetime.now().isoformat(),
        "git_head": git_info.get("head", ""),
        "git_dirty": git_info.get("dirty", ""),
        "train_config": (
            OmegaConf.to_yaml(system.training_config, resolve=True)
            if system.training_config is not None
            else ""
        ),
        "results_section": "" if minimal else _resolve_results_section(scores_path),
    }
    if not minimal:
        context.update(dict(getattr(pack_cfg, "readme_context", {}) or {}))
    readme_text = _render_readme(readme_template, context)
    (out_dir / "README.md").write_text(readme_text, encoding="utf-8")


def _write_meta(
    *,
    out_dir: Path,
    files: Dict[str, str] | None = None,
    yaml_files: Dict[str, str] | None = None,
    extra_fields: Dict[str, Any] | None = None,
) -> None:
    """Write meta.yaml into output directory."""
    meta = _build_meta(
        files=files or {},
        yaml_files=yaml_files or {},
        extra_fields=extra_fields or {},
    )
    (out_dir / "meta.yaml").write_text(OmegaConf.to_yaml(meta), encoding="utf-8")


def _build_meta(
    files: Dict[str, str],
    yaml_files: Dict[str, str],
    extra_fields: Dict[str, Any] | None = None,
) -> dict:
    """Build espnet2-style meta.yaml contents."""
    meta = {
        "files": dict(files),
        "yaml_files": dict(yaml_files),
        "timestamp": datetime.now().timestamp(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "espnet": str(espnet2.__version__),
    }
    meta.update(extra_fields or {})
    return meta


def _resolve_recipe_root(system) -> Path | None:
    """Return recipe root when available from the system configs."""
    for cfg_name in ("training_config", "inference_config", "metrics_config"):
        cfg = getattr(system, cfg_name, None)
        recipe_dir = getattr(cfg, "recipe_dir", None) if cfg is not None else None
        if recipe_dir:
            path = Path(recipe_dir).resolve()
            if path.exists():
                return path
    return None


def _relative_to_root(path: Path, root: Path | None) -> Path | None:
    """Return ``path`` relative to ``root`` when possible."""
    if root is None:
        return None
    try:
        return path.absolute().relative_to(root.resolve())
    except ValueError:
        return None


def _copy_path(
    *,
    src: Path,
    dst: Path,
    ignore=None,
) -> None:
    """Copy a file or directory into the bundle."""
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _default_copy_dest(
    *,
    src: Path,
    out_dir: Path,
    exp_dir: Path,
    recipe_root: Path | None,
) -> Path:
    """Return bundle destination for a copied include/extra path."""
    relative = _relative_to_root(src, recipe_root)
    if relative is not None:
        return out_dir / relative
    if src.resolve() == exp_dir.resolve():
        return out_dir / "exp"
    return out_dir / src.name


def _artifact_dest(
    *,
    src: Path,
    out_dir: Path,
    recipe_root: Path | None,
    key: str,
) -> Path:
    """Return bundle destination for a manifest entry file."""
    relative = _relative_to_root(src, recipe_root)
    if relative is not None:
        return out_dir / relative
    suffix = "".join(src.suffixes) or src.suffix
    return out_dir / "artifacts" / f"{key}{suffix}"


def _bundle_config_copy(config: DictConfig) -> DictConfig:
    """Return a config copy rewritten for bundle-relative use."""
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    if getattr(cfg, "recipe_dir", None) not in (None, ""):
        cfg.recipe_dir = "."
    return cfg


def _is_path_like_key(key: str | None) -> bool:
    """Return whether ``key`` usually stores a filesystem path."""
    if not key:
        return False
    if key == "recipe_dir":
        return False
    return key in _PATH_LIKE_KEYS or key.endswith(("_dir", "_path", "_file"))


def _resolve_bundle_yaml_reference(
    raw_path: str,
    *,
    base_dir: Path,
    out_dir: Path,
) -> Path | None:
    """Return a bundle-local YAML path for a config reference when possible."""
    normalized = _strip_optional_quotes(raw_path.strip())
    if not normalized:
        return None
    if normalized == "${recipe_dir}":
        candidate = out_dir
    elif normalized.startswith("${recipe_dir}/"):
        candidate = out_dir / normalized[len("${recipe_dir}/") :]
    elif "${" in normalized:
        return None
    else:
        candidate = Path(normalized)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).absolute()
        else:
            candidate = candidate.absolute()

    if candidate.suffix.lower() not in _YAML_SUFFIXES:
        return None
    return candidate


def _iter_yaml_references(
    value: Any,
    *,
    base_dir: Path,
    out_dir: Path,
    key: str | None = None,
):
    """Yield bundle-local YAML references discovered in a config value tree."""
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _iter_yaml_references(
                child_value,
                base_dir=base_dir,
                out_dir=out_dir,
                key=str(child_key),
            )
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_yaml_references(
                item,
                base_dir=base_dir,
                out_dir=out_dir,
                key=key,
            )
        return
    if not isinstance(value, str):
        return

    if _is_path_like_key(key):
        candidate = _resolve_bundle_yaml_reference(
            value,
            base_dir=base_dir,
            out_dir=out_dir,
        )
        if candidate is not None:
            yield candidate

    for match in _LOAD_YAML_PATTERN.finditer(value):
        candidate = _resolve_bundle_yaml_reference(
            match.group("path"),
            base_dir=base_dir,
            out_dir=out_dir,
        )
        if candidate is not None:
            yield candidate


def _enqueue_yaml_reference(
    candidate: Path,
    *,
    out_dir: Path,
    keep_paths: set[Path],
    queue: list[Path],
) -> None:
    """Register a bundle-local YAML path and queue it for recursive scanning."""
    bundle_root = out_dir.resolve()
    normalized_candidates = [candidate.absolute()]
    if candidate.exists():
        normalized_candidates.append(candidate.resolve())

    for normalized in normalized_candidates:
        try:
            normalized.relative_to(bundle_root)
        except ValueError:
            continue
        if not normalized.exists() or not normalized.is_file():
            continue
        if normalized in keep_paths:
            continue
        keep_paths.add(normalized)
        if (
            normalized.name != "meta.yaml"
            and normalized.suffix.lower() in _YAML_SUFFIXES
        ):
            queue.append(normalized)


def _prune_unreferenced_bundle_yaml_files(out_dir: Path) -> None:
    """Remove bundle YAML files that are unreachable from ``meta.yaml`` roots."""
    meta_path = out_dir / "meta.yaml"
    if not meta_path.is_file():
        return

    bundle_root = out_dir.resolve()
    keep_paths: set[Path] = {meta_path.resolve()}
    queue: list[Path] = []

    meta = OmegaConf.to_container(OmegaConf.load(meta_path), resolve=False) or {}
    if not isinstance(meta, dict):
        raise RuntimeError(f"Expected mapping metadata in {meta_path}")

    root_entries = []
    for section_name in ("yaml_files", "files"):
        section = meta.get(section_name) or {}
        if isinstance(section, dict):
            root_entries.extend(section.values())
    if (out_dir / "conf" / "inference.yaml").is_file():
        root_entries.append("conf/inference.yaml")

    for entry in root_entries:
        if not isinstance(entry, str):
            continue
        candidate = _resolve_bundle_yaml_reference(
            entry,
            base_dir=out_dir,
            out_dir=out_dir,
        )
        if candidate is None:
            continue
        _enqueue_yaml_reference(
            candidate,
            out_dir=out_dir,
            keep_paths=keep_paths,
            queue=queue,
        )

    while queue:
        current = queue.pop()
        loaded = OmegaConf.to_container(OmegaConf.load(current), resolve=False)
        for candidate in _iter_yaml_references(
            loaded,
            base_dir=current.parent,
            out_dir=out_dir,
        ):
            _enqueue_yaml_reference(
                candidate,
                out_dir=out_dir,
                keep_paths=keep_paths,
                queue=queue,
            )

    pruned = 0
    for path in out_dir.rglob("*"):
        if path.name == "meta.yaml" or path.suffix.lower() not in _YAML_SUFFIXES:
            continue
        normalized = path.resolve()
        if normalized in keep_paths or path.absolute() in keep_paths:
            continue
        path.unlink()
        pruned += 1

    if pruned:
        logger.info("Pruned %d unreferenced YAML files from %s", pruned, bundle_root)

    for path in sorted(
        out_dir.rglob("*"), key=lambda candidate: len(candidate.parts), reverse=True
    ):
        if not path.is_dir():
            continue
        try:
            path.rmdir()
        except OSError:
            continue


def _to_bundle_relative_path(value: str, recipe_root: Path | None) -> str:
    """Rewrite a path string so it resolves from ``recipe_dir`` inside a bundle."""
    if not value or "${" in value or recipe_root is None:
        return value

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (recipe_root / candidate).absolute()
    else:
        candidate = candidate.absolute()

    relative = _relative_to_root(candidate, recipe_root)
    if relative is None:
        return value
    if relative.as_posix() == ".":
        return "${recipe_dir}"
    return f"${{recipe_dir}}/{relative.as_posix()}"


def _rewrite_bundle_paths(value: Any, recipe_root: Path | None, key: str | None = None):
    """Recursively rewrite path-like config values to bundle-relative paths."""
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=False)
    if isinstance(value, ListConfig):
        value = OmegaConf.to_container(value, resolve=False)
    if isinstance(value, dict):
        return {
            child_key: _rewrite_bundle_paths(child_value, recipe_root, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_bundle_paths(item, recipe_root, key) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_bundle_paths(item, recipe_root, key) for item in value)
    if isinstance(value, str) and _is_path_like_key(key):
        return _to_bundle_relative_path(value, recipe_root)
    return value


def _bundle_runtime_config(config: DictConfig, recipe_root: Path | None) -> DictConfig:
    """Return a bundle-ready config with recipe-root-relative paths."""
    cfg = _bundle_config_copy(config)
    plain_cfg = OmegaConf.to_container(cfg, resolve=False)
    rewritten = _rewrite_bundle_paths(plain_cfg, recipe_root)
    bundle_cfg = OmegaConf.create(rewritten)
    if isinstance(bundle_cfg, DictConfig) and "recipe_dir" in bundle_cfg:
        bundle_cfg.recipe_dir = "."
    return bundle_cfg


def _write_embedded_configs(
    system, out_dir: Path, recipe_root: Path | None
) -> Dict[str, str]:
    """Write in-memory configs into ``conf/`` when recipe files are unavailable."""
    from espnet3.utils.config_utils import (
        load_and_merge_config,
        load_config_with_defaults,
    )

    conf_dir = out_dir / "conf"
    entries = {
        "training_config": ("training.yaml", getattr(system, "training_config", None)),
        "inference_config": (
            "inference.yaml",
            getattr(system, "inference_config", None),
        ),
        "metrics_config": ("metrics.yaml", getattr(system, "metrics_config", None)),
        "publication_config": (
            "publication.yaml",
            getattr(system, "publication_config", None),
        ),
    }
    written: Dict[str, str] = {}
    for meta_key, (filename, config) in entries.items():
        if config is None and recipe_root is not None:
            candidate = recipe_root / "conf" / filename
            if candidate.is_file():
                if meta_key == "publication_config":
                    config = load_config_with_defaults(str(candidate))
                else:
                    config = load_and_merge_config(candidate, config_name=filename)
        if config is None:
            continue
        conf_dir.mkdir(parents=True, exist_ok=True)
        bundle_cfg = _bundle_runtime_config(config, recipe_root)
        target = conf_dir / filename
        target.write_text(OmegaConf.to_yaml(bundle_cfg), encoding="utf-8")
        written[meta_key] = target.relative_to(out_dir).as_posix()
    return written


def _resolve_pack_task(system, pack_cfg: DictConfig) -> str:
    """Infer the publication task name used for default artifact keys."""
    espnet2_cfg = getattr(pack_cfg, "espnet2", None)
    task = getattr(espnet2_cfg, "task", None)
    if task:
        return str(task)
    training_task = str(
        getattr(getattr(system, "training_config", None), "task", "") or ""
    )
    if ".asr." in training_task or training_task.endswith(".ASRTask"):
        return "asr"
    return ""


def _resolve_default_artifacts(system, pack_cfg: DictConfig) -> dict[str, Any]:
    """Return default bundle artifacts copied for direct inference use."""
    files: Dict[str, str] = {}
    yaml_files: Dict[str, str] = {}
    copy_paths: list[Path] = []

    training_cfg = getattr(system, "training_config", None)
    if training_cfg is None:
        return {"files": files, "yaml_files": yaml_files, "copy_paths": copy_paths}

    exp_dir = Path(training_cfg.exp_dir)
    data_dir = getattr(training_cfg, "data_dir", None)
    include_data_dir = bool(getattr(pack_cfg, "include_data_dir", True))
    if data_dir and include_data_dir:
        data_path = Path(data_dir)
        if data_path.exists():
            copy_paths.append(data_path)

    task = _resolve_pack_task(system, pack_cfg)
    if task == "asr":
        train_config = exp_dir / "config.yaml"
        model_file = exp_dir / "last.ckpt"
        if train_config.exists():
            yaml_files.setdefault("asr_train_config", str(train_config))
        if model_file.exists():
            files.setdefault("asr_model_file", str(model_file))

    return {"files": files, "yaml_files": yaml_files, "copy_paths": copy_paths}


def _copy_recipe_assets(
    *,
    system,
    out_dir: Path,
    ignore,
) -> dict[str, Any]:
    """Copy recipe-local assets needed for config-based inference."""
    recipe_root = _resolve_recipe_root(system)
    yaml_files: Dict[str, str] = {}
    user_code_paths: list[str] = []

    if recipe_root is None:
        yaml_files.update(_write_embedded_configs(system, out_dir, recipe_root))
    else:
        for name in _RECIPE_BUNDLE_ENTRIES:
            src = recipe_root / name
            if not src.exists():
                continue
            dst = out_dir / name
            _copy_path(src=src, dst=dst, ignore=ignore)
        yaml_files.update(_write_embedded_configs(system, out_dir, recipe_root))

        if (out_dir / "conf" / "inference.yaml").is_file():
            yaml_files["inference_config"] = "conf/inference.yaml"
        if (out_dir / "conf" / "training.yaml").is_file():
            yaml_files["training_config"] = "conf/training.yaml"
        if (out_dir / "conf" / "metrics.yaml").is_file():
            yaml_files["metrics_config"] = "conf/metrics.yaml"
        if (out_dir / "conf" / "publication.yaml").is_file():
            yaml_files["publication_config"] = "conf/publication.yaml"

    if (out_dir / "src").exists():
        user_code_paths.append("src")

    return {
        "recipe_root": recipe_root,
        "yaml_files": yaml_files,
        "extra_fields": {"user_code_paths": user_code_paths} if user_code_paths else {},
    }


def _copy_manifest_entries(
    *,
    pack_cfg: DictConfig,
    out_dir: Path,
    recipe_root: Path | None,
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Copy declared manifest files into the bundle and return relative paths."""
    file_entries = dict(getattr(pack_cfg, "files", {}) or {})
    yaml_entries = dict(getattr(pack_cfg, "yaml_files", {}) or {})
    return _copy_artifact_entries(
        file_entries=file_entries,
        yaml_entries=yaml_entries,
        out_dir=out_dir,
        recipe_root=recipe_root,
    )


def _copy_artifact_entries(
    *,
    file_entries: Dict[str, str],
    yaml_entries: Dict[str, str],
    out_dir: Path,
    recipe_root: Path | None,
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Copy artifact files into the bundle and return relative manifest paths."""
    copied_files: Dict[str, str] = {}
    copied_yaml_files: Dict[str, str] = {}

    for key, raw_path in file_entries.items():
        src = Path(raw_path)
        if not src.exists():
            raise RuntimeError(f"pack_model.files entry does not exist: {src}")
        dst = _artifact_dest(
            src=src, out_dir=out_dir, recipe_root=recipe_root, key=str(key)
        )
        _copy_path(src=src, dst=dst)
        copied_files[str(key)] = dst.relative_to(out_dir).as_posix()

    for key, raw_path in yaml_entries.items():
        src = Path(raw_path)
        if not src.exists():
            raise RuntimeError(f"pack_model.yaml_files entry does not exist: {src}")
        dst = _artifact_dest(
            src=src, out_dir=out_dir, recipe_root=recipe_root, key=str(key)
        )
        _copy_path(src=src, dst=dst)
        copied_yaml_files[str(key)] = dst.relative_to(out_dir).as_posix()

    return copied_files, copied_yaml_files


def _merge_meta_file(
    *,
    out_dir: Path,
    files: Dict[str, str] | None = None,
    yaml_files: Dict[str, str] | None = None,
    extra_fields: Dict[str, Any] | None = None,
) -> None:
    """Merge new metadata entries into an existing ``meta.yaml`` file."""
    meta_path = out_dir / "meta.yaml"
    existing = OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True) or {}
    if not isinstance(existing, dict):
        raise RuntimeError(f"Expected mapping metadata in {meta_path}")
    merged_files = dict(existing.get("files") or {})
    merged_files.update(files or {})
    merged_yaml_files = dict(existing.get("yaml_files") or {})
    merged_yaml_files.update(yaml_files or {})
    existing["files"] = merged_files
    existing["yaml_files"] = merged_yaml_files
    existing.update(extra_fields or {})
    meta_path.write_text(OmegaConf.to_yaml(existing), encoding="utf-8")


def _render_readme(template_text: str, context: Dict[str, str]) -> str:
    """Render README template with string.Template."""
    raw = template_text
    safe_context = {k: "" if v is None else str(v) for k, v in context.items()}
    lines = []
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
    for line in raw.splitlines():
        placeholders = pattern.findall(line)
        if placeholders and any(not safe_context.get(key, "") for key in placeholders):
            continue
        lines.append(line)
    template = Template("\n".join(lines))
    return template.safe_substitute(safe_context)


def _resolve_results_section(scores_path: Optional[Path]) -> str:
    """Render a results table from scores.json.

    Args:
        scores_path: Optional path to scores.json.
    Returns:
        Markdown table string; empty string when scores are missing or invalid.
    Note:
        The table has one row per test set and metric names as columns.
    """
    if scores_path is None or not scores_path.exists():
        return ""
    try:
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    metrics_by_test: dict[str, dict[str, str]] = {}
    metric_keys: set[str] = set()
    for per_test in scores.values():
        if not isinstance(per_test, dict):
            continue
        for test_name, metrics in per_test.items():
            if not isinstance(metrics, dict):
                continue
            test_key = str(test_name)
            metrics_by_test.setdefault(test_key, {})
            for metric_name, value in metrics.items():
                metric_key = str(metric_name)
                metric_keys.add(metric_key)
                metrics_by_test[test_key][metric_key] = str(value)
    if not metrics_by_test or not metric_keys:
        return ""
    ordered_metrics = sorted(metric_keys)
    header = "| Test | " + " | ".join(ordered_metrics) + " |"
    separator = "| --- | " + " | ".join("---" for _ in ordered_metrics) + " |"
    lines = ["## Results", "", header, separator]
    for test_name in sorted(metrics_by_test.keys()):
        values = [metrics_by_test[test_name].get(m, "") for m in ordered_metrics]
        lines.append("| " + " | ".join([test_name] + values) + " |")
    lines.append("")
    return "\n".join(lines)


def _git_info() -> Dict[str, str]:
    """Return git head/dirtiness if available."""
    if shutil.which("git") is None:
        return {"head": "", "dirty": ""}
    try:
        head = _run(["git", "rev-parse", "HEAD"])
        dirty = "dirty" if _run(["git", "status", "--porcelain"]) else "clean"
        return {"head": head, "dirty": dirty}
    except Exception:
        return {"head": "", "dirty": ""}


def _hf_username() -> str:
    """Return Hugging Face username from huggingface-cli if available."""
    if shutil.which("huggingface-cli") is None:
        return ""
    try:
        output = _run(["huggingface-cli", "whoami"])
    except Exception:
        return ""
    lines = [line.strip() for line in output.splitlines()]
    for idx, stripped in enumerate(lines):
        if stripped.lower().startswith("orgs:") and idx > 0:
            candidate = lines[idx - 1].strip()
            return candidate or ""
    return ""


def _resolve_recipe(system) -> str:
    """Return egs3/<dataset>/<system> from run.py location if possible."""
    candidate_paths = []
    recipe_cfg = getattr(getattr(system, "training_config", None), "recipe", None)
    if recipe_cfg:
        return str(recipe_cfg)
    argv0 = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if argv0 is not None:
        candidate_paths.append(argv0)
    recipe_dir = getattr(getattr(system, "training_config", None), "recipe_dir", None)
    if recipe_dir:
        candidate_paths.append(Path(recipe_dir) / "run.py")
    for path in candidate_paths:
        parts = list(Path(path).resolve().parts)
        if "egs3" in parts:
            idx = parts.index("egs3")
            if len(parts) >= idx + 3:
                return "/".join(parts[idx + 1 : idx + 3])
    return ""


def pack_model(
    system,
    *,
    include: Optional[List[Path]] = None,
    extra: Optional[List[Path]] = None,
) -> Path:
    """Pack model artifacts for publishing.

    Attributes:
        system: ESPnet3 system instance providing configs and helper paths.
    Args:
        system: Object with ``training_config`` and optional
            ``publication_config``.
        include: Optional explicit include paths
            (defaults to exp_dir and config includes).
        extra: Optional explicit extra paths (defaults to config extras).
    Returns:
        Path to the packed output directory.
    Yields:
        None.
    Raises:
        RuntimeError: If ``training_config`` is missing or required files are
            absent.
    Examples:
        >>> from espnet3.utils.publish import pack_model
        >>> pack_path = pack_model(system)
        >>> print(pack_path)
    Note:
        ``pack_model.strategy`` controls whether espnet2 or espnet3 packing
        logic is used. The output location can be set via
        ``publication_config.pack_model.out_dir``.
        ``publication_config.pack_model.readme_template`` can override the
        default
        README template path.
        ``publication_config.pack_model.decode_dir`` can be used to specify
        which scores.json to include. If not set, it falls back to
        ``inference_config.decode_dir``.
        README generation uses scores.json to render a per-test metrics table.
    Todo:
        Support more configurable exclusions if needed.
    """
    if system.training_config is None:
        raise RuntimeError("pack_model requires training_config.")

    exp_dir = Path(system.training_config.exp_dir)
    publication_cfg = getattr(system, "publication_config", None)
    if publication_cfg is None:
        raise RuntimeError(
            "pack_model requires publication_config " "(publication_config.pack_model)."
        )
    pack_cfg = getattr(publication_cfg, "pack_model", None)
    if pack_cfg is None:
        raise RuntimeError("pack_model requires publication_config.pack_model.")

    if getattr(getattr(system, "training_config", None), "task", None):
        strategy = "espnet2"
    else:
        strategy = "espnet3"

    readme_template = _load_readme_template(pack_cfg)

    out_dir = Path(getattr(pack_cfg, "out_dir", exp_dir / "model_pack"))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_includes = (
        [] if strategy == "espnet2" else [Path(system.training_config.exp_dir)]
    )
    include_paths = list(include) if include is not None else list(default_includes)
    include_cfg = getattr(pack_cfg, "include", None)
    if include_cfg:
        if isinstance(include_cfg, (list, tuple, ListConfig)):
            include_paths += [Path(p) for p in include_cfg]
        else:
            include_paths.append(Path(include_cfg))

    extra_paths = list(extra) if extra is not None else []
    extra_cfg = getattr(pack_cfg, "extra", None)
    if extra_cfg:
        if isinstance(extra_cfg, (list, tuple, ListConfig)):
            extra_paths += [Path(p) for p in extra_cfg]
        else:
            extra_paths.append(Path(extra_cfg))

    default_artifacts = _resolve_default_artifacts(system, pack_cfg)
    extra_paths.extend(default_artifacts["copy_paths"])

    excludes = []
    exclude_cfg = getattr(pack_cfg, "exclude", None)
    if exclude_cfg:
        if isinstance(exclude_cfg, (list, tuple, ListConfig)):
            excludes += [str(p) for p in exclude_cfg]
        else:
            excludes.append(str(exclude_cfg))

    effective = list(excludes)
    effective.extend(["decode", "decode_*", "decode_dir"])
    if out_dir.name:
        effective.append(out_dir.name)
    effective.append("__pycache__")

    def _normalize(pattern: str) -> str:
        return pattern.replace("**/", "").replace("/**", "")

    ignore = shutil.ignore_patterns(*[_normalize(str(p)) for p in effective])

    if strategy == "espnet2":
        spec = _resolve_espnet2_spec(pack_cfg)
        out_path = out_dir / "model_pack.zip"
        espnet2_pack(
            files={str(k): str(v) for k, v in spec["files"].items()},
            yaml_files={str(k): str(v) for k, v in spec["yaml_files"].items()},
            option=[str(p) for p in spec["option"]],
            outpath=out_path,
        )
        shutil.unpack_archive(out_path, out_dir)
        out_path.unlink()
        recipe_bundle = _copy_recipe_assets(
            system=system, out_dir=out_dir, ignore=ignore
        )
        recipe_root = recipe_bundle["recipe_root"]
        for path in include_paths:
            if not path.exists():
                logger.warning("Pack include path does not exist: %s", path)
                continue
            dest = _default_copy_dest(
                src=path,
                out_dir=out_dir,
                exp_dir=exp_dir,
                recipe_root=recipe_root,
            )
            _copy_path(src=path, dst=dest, ignore=ignore)

        for path in extra_paths:
            if not path.exists():
                logger.warning("Pack extra path does not exist: %s", path)
                continue
            dest = _default_copy_dest(
                src=path,
                out_dir=out_dir,
                exp_dir=exp_dir,
                recipe_root=recipe_root,
            )
            _copy_path(src=path, dst=dest, ignore=ignore)

        copied_default_files, copied_default_yaml_files = _copy_artifact_entries(
            file_entries=default_artifacts["files"],
            yaml_entries=default_artifacts["yaml_files"],
            out_dir=out_dir,
            recipe_root=recipe_root,
        )

        decode_dir = getattr(pack_cfg, "decode_dir", None)
        if decode_dir is None and getattr(system, "inference_config", None) is not None:
            decode_dir = getattr(system.inference_config, "decode_dir", None)
        resolved_scores = None
        if decode_dir:
            matches = list(Path(decode_dir).rglob("scores.json"))
            if len(matches) > 1:
                raise RuntimeError(
                    "Multiple scores.json found; set "
                    "publication_config.pack_model.decode_dir to a single "
                    "decode directory."
                )
            if len(matches) == 1:
                resolved_scores = matches[0]
        if resolved_scores is not None and resolved_scores.exists():
            shutil.copy2(resolved_scores, out_dir / "scores.json")

        _write_readme(
            readme_template=readme_template,
            out_dir=out_dir,
            publication_cfg=publication_cfg,
            pack_cfg=pack_cfg,
            exp_dir=exp_dir,
            strategy=strategy,
            system=system,
            scores_path=resolved_scores,
        )
        _merge_meta_file(
            out_dir=out_dir,
            files=copied_default_files,
            yaml_files={**copied_default_yaml_files, **recipe_bundle["yaml_files"]},
            extra_fields=recipe_bundle["extra_fields"],
        )
        _prune_unreferenced_bundle_yaml_files(out_dir)
        logger.info("Packed model (espnet2) to %s", out_dir)
        return out_dir
    recipe_bundle = _copy_recipe_assets(system=system, out_dir=out_dir, ignore=ignore)
    recipe_root = recipe_bundle["recipe_root"]

    for path in include_paths:
        if not path.exists():
            logger.warning("Pack include path does not exist: %s", path)
            continue
        dest = _default_copy_dest(
            src=path,
            out_dir=out_dir,
            exp_dir=exp_dir,
            recipe_root=recipe_root,
        )
        _copy_path(src=path, dst=dest, ignore=ignore)

    for path in extra_paths:
        if not path.exists():
            logger.warning("Pack extra path does not exist: %s", path)
            continue
        dest = _default_copy_dest(
            src=path,
            out_dir=out_dir,
            exp_dir=exp_dir,
            recipe_root=recipe_root,
        )
        _copy_path(src=path, dst=dest, ignore=ignore)

    decode_dir = getattr(pack_cfg, "decode_dir", None)
    if decode_dir is None and getattr(system, "inference_config", None) is not None:
        decode_dir = getattr(system.inference_config, "decode_dir", None)
    resolved_scores = None
    if decode_dir:
        matches = list(Path(decode_dir).rglob("scores.json"))
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple scores.json found; set "
                "publication_config.pack_model.decode_dir to a single "
                "decode directory."
            )
        if len(matches) == 1:
            resolved_scores = matches[0]
    if resolved_scores is not None and resolved_scores.exists():
        shutil.copy2(resolved_scores, out_dir / "scores.json")
    else:
        logger.info("No scores.json found; skipping root copy.")

    _write_readme(
        readme_template=readme_template,
        out_dir=out_dir,
        publication_cfg=publication_cfg,
        pack_cfg=pack_cfg,
        exp_dir=exp_dir,
        strategy=strategy,
        system=system,
        scores_path=resolved_scores,
    )
    copied_files, copied_yaml_files = _copy_manifest_entries(
        pack_cfg=pack_cfg,
        out_dir=out_dir,
        recipe_root=recipe_root,
    )
    copied_default_files, copied_default_yaml_files = _copy_artifact_entries(
        file_entries=default_artifacts["files"],
        yaml_entries=default_artifacts["yaml_files"],
        out_dir=out_dir,
        recipe_root=recipe_root,
    )
    copied_files.update(copied_default_files)
    copied_yaml_files.update(copied_default_yaml_files)
    copied_yaml_files.update(recipe_bundle["yaml_files"])
    _write_meta(
        out_dir=out_dir,
        files=copied_files,
        yaml_files=copied_yaml_files,
        extra_fields=recipe_bundle["extra_fields"],
    )
    _prune_unreferenced_bundle_yaml_files(out_dir)

    logger.info("Packed model to %s", out_dir)
    return out_dir


def _upload_common(
    repo: str,
    src_dir: Path,
    *,
    repo_type: str,
    create_options: Optional[dict] = None,
    create_repo_name: Optional[str] = None,
) -> None:
    """Upload artifacts to a Hugging Face repo via huggingface-cli."""
    if shutil.which("huggingface-cli") is None:
        raise RuntimeError("huggingface-cli is required for upload.")

    create_options = dict(create_options or {})
    exists = _repo_exists(repo, repo_type)
    if exists is False or exists is None:
        _create_repo(
            create_repo_name or repo,
            repo_type=repo_type,
            organization=create_options.get("organization"),
            space_sdk=create_options.get("space_sdk"),
            yes=create_options.get("yes", True),
        )

    upload_cmd = [
        "huggingface-cli",
        "upload",
        repo,
        str(src_dir),
        "--repo-type",
        repo_type,
    ]
    _run(upload_cmd)


def upload_model(system) -> None:
    """Upload packed model artifacts to a Hugging Face model repo.

    Attributes:
        system: ESPnet3 system instance providing configs and helper paths.
    Args:
        system: Object with ``publication_config`` and optionally
            ``training_config``.
    Returns:
        None.
    Yields:
        None.
    Raises:
        RuntimeError: If required config values are missing or upload fails.
    Examples:
        >>> from espnet3.utils.publish import upload_model
        >>> upload_model(system)
    Note:
        Uploads the pack directory for both espnet2/espnet3.
    Todo:
        Add optional README or metadata generation if needed.
    """
    publication_cfg = getattr(system, "publication_config", None)
    if publication_cfg is None:
        raise RuntimeError(
            "upload_model requires publication_config "
            "(publication_config.upload_model)."
        )
    upload_cfg = getattr(publication_cfg, "upload_model", None)
    if upload_cfg is None:
        raise RuntimeError("upload_model requires publication_config.upload_model.")
    repo = getattr(upload_cfg, "hf_repo", None)
    if not repo:
        raise RuntimeError(
            "upload_model requires publication_config.upload_model.hf_repo"
        )

    exp_dir = (
        Path(system.training_config.exp_dir) if system.training_config else Path.cwd()
    )
    pack_cfg = getattr(publication_cfg, "pack_model", None) or OmegaConf.create({})
    pack_dir = Path(getattr(pack_cfg, "out_dir", exp_dir / "model_pack"))
    if not pack_dir.exists():
        raise RuntimeError(f"Model pack not found: {pack_dir}")

    _upload_common(
        repo,
        pack_dir,
        repo_type="model",
    )
