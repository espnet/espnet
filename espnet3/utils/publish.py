"""Publication helpers for ESPnet3."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from string import Template
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import espnet2
import torch
from omegaconf import DictConfig, OmegaConf
from espnet2.main_funcs.pack_funcs import pack as espnet2_pack

logger = logging.getLogger(__name__)


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
    publish_cfg: DictConfig,
    pack_cfg: DictConfig,
    exp_dir: Path,
    strategy: str,
    system,
    scores_path: Optional[Path],
) -> None:
    """Render README template into output directory if present."""
    if not readme_template:
        return
    git_info = _git_info()
    context = {
        "hf_repo": getattr(getattr(publish_cfg, "upload_model", None), "hf_repo", ""),
        "system": system.__class__.__name__,
        "recipe": _resolve_recipe(system),
        "creator": _hf_username(),
        "pack_name": out_dir.name,
        "pack_strategy": strategy,
        "exp_dir": str(exp_dir),
        "created_at": datetime.now().isoformat(),
        "git_head": git_info.get("head", ""),
        "git_dirty": git_info.get("dirty", ""),
        "train_config": (
            OmegaConf.to_yaml(system.train_config, resolve=True)
            if system.train_config is not None
            else ""
        ),
        "results_section": _resolve_results_section(scores_path),
    }
    context.update(dict(getattr(pack_cfg, "readme_context", {}) or {}))
    readme_text = _render_readme(readme_template, context)
    (out_dir / "README.md").write_text(readme_text, encoding="utf-8")


def _write_meta(pack_cfg: DictConfig, out_dir: Path) -> None:
    """Write meta.yaml into output directory."""
    files_cfg = dict(getattr(pack_cfg, "files", {}) or {})
    yaml_files_cfg = dict(getattr(pack_cfg, "yaml_files", {}) or {})
    meta = _build_meta(
        files={str(k): Path(v) for k, v in files_cfg.items()},
        yaml_files={str(k): Path(v) for k, v in yaml_files_cfg.items()},
    )
    (out_dir / "meta.yaml").write_text(OmegaConf.to_yaml(meta), encoding="utf-8")


def _build_meta(files: Dict[str, Path], yaml_files: Dict[str, Path]) -> dict:
    """Build espnet2-style meta.yaml contents."""
    meta = {
        "files": {k: str(Path(v).resolve()) for k, v in files.items()},
        "yaml_files": {k: str(Path(v).resolve()) for k, v in yaml_files.items()},
        "timestamp": datetime.now().timestamp(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "espnet": str(espnet2.__version__),
    }
    return meta


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
    recipe_cfg = getattr(getattr(system, "train_config", None), "recipe", None)
    if recipe_cfg:
        return str(recipe_cfg)
    argv0 = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if argv0 is not None:
        candidate_paths.append(argv0)
    recipe_dir = getattr(getattr(system, "train_config", None), "recipe_dir", None)
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
        system: Object with ``train_config`` and optional ``publish_config``.
        include: Optional explicit include paths (defaults to exp_dir and config includes).
        extra: Optional explicit extra paths (defaults to config extras).
    Returns:
        Path to the packed output directory.
    Yields:
        None.
    Raises:
        RuntimeError: If ``train_config`` is missing or required files are absent.
    Examples:
        >>> from espnet3.utils.publish import pack_model
        >>> pack_path = pack_model(system)
        >>> print(pack_path)
    Note:
        ``pack_model.strategy`` controls whether espnet2 or espnet3 packing
        logic is used. The output location can be set via
        ``publish.pack_model.out_dir``.
        ``publish.pack_model.readme_template`` can override the default
        README template path.
        ``publish.pack_model.decode_dir`` can be used to specify which scores.json
        to include. If not set, it falls back to ``infer_config.decode_dir``.
        README generation uses scores.json to render a per-test metrics table.
    Todo:
        Support more configurable exclusions if needed.
    """
    if system.train_config is None:
        raise RuntimeError("pack_model requires train_config.")

    exp_dir = Path(system.train_config.exp_dir)
    if system.publish_config is None:
        raise RuntimeError(
            "pack_model requires publish_config (publish_config.pack_model)."
        )
    publish_cfg = system.publish_config
    pack_cfg = getattr(publish_cfg, "pack_model", None)
    if pack_cfg is None:
        raise RuntimeError("pack_model requires publish_config.pack_model.")

    if getattr(getattr(system, "train_config", None), "task", None):
        strategy = "espnet2"
    else:
        strategy = "espnet3"

    readme_template = _load_readme_template(pack_cfg)

    out_dir = Path(getattr(pack_cfg, "out_dir", exp_dir / "model_pack"))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

        decode_dir = getattr(pack_cfg, "decode_dir", None)
        if decode_dir is None and getattr(system, "infer_config", None) is not None:
            decode_dir = getattr(system.infer_config, "decode_dir", None)
        resolved_scores = None
        if decode_dir:
            matches = list(Path(decode_dir).rglob("scores.json"))
            if len(matches) > 1:
                raise RuntimeError(
                    "Multiple scores.json found; set publish.pack_model.decode_dir "
                    "to a single decode directory."
                )
            if len(matches) == 1:
                resolved_scores = matches[0]
        if resolved_scores is not None and resolved_scores.exists():
            shutil.copy2(resolved_scores, out_dir / "scores.json")

        _write_readme(
            readme_template=readme_template,
            out_dir=out_dir,
            publish_cfg=publish_cfg,
            pack_cfg=pack_cfg,
            exp_dir=exp_dir,
            strategy=strategy,
            system=system,
            scores_path=resolved_scores,
        )
        logger.info("Packed model (espnet2) to %s", out_dir)
        return out_dir

    include_paths = (
        list(include) if include is not None else [Path(system.train_config.exp_dir)]
    )
    include_cfg = getattr(pack_cfg, "include", None)
    if include_cfg:
        if isinstance(include_cfg, (list, tuple)):
            include_paths += [Path(p) for p in include_cfg]
        else:
            include_paths.append(Path(include_cfg))

    extra_paths = list(extra) if extra is not None else []
    extra_cfg = getattr(pack_cfg, "extra", None)
    if extra_cfg:
        if isinstance(extra_cfg, (list, tuple)):
            extra_paths += [Path(p) for p in extra_cfg]
        else:
            extra_paths.append(Path(extra_cfg))

    excludes = []
    exclude_cfg = getattr(pack_cfg, "exclude", None)
    if exclude_cfg:
        if isinstance(exclude_cfg, (list, tuple)):
            excludes += [str(p) for p in exclude_cfg]
        else:
            excludes.append(str(exclude_cfg))

    effective = list(excludes)
    effective.extend(["decode", "decode_*", "decode_dir", "publish"])
    if out_dir.name:
        effective.append(out_dir.name)

    def _normalize(pattern: str) -> str:
        return pattern.replace("**/", "").replace("/**", "")

    ignore = shutil.ignore_patterns(*[_normalize(str(p)) for p in effective])

    for path in include_paths:
        if not path.exists():
            logger.warning("Pack include path does not exist: %s", path)
            continue
        dest = out_dir / ("exp" if path.resolve() == exp_dir.resolve() else path.name)
        if path.is_dir():
            shutil.copytree(path, dest, ignore=ignore, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)

    for path in extra_paths:
        if not path.exists():
            logger.warning("Pack extra path does not exist: %s", path)
            continue
        dest = out_dir / ("exp" if path.resolve() == exp_dir.resolve() else path.name)
        if path.is_dir():
            shutil.copytree(path, dest, ignore=ignore, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)

    decode_dir = getattr(pack_cfg, "decode_dir", None)
    if decode_dir is None and getattr(system, "infer_config", None) is not None:
        decode_dir = getattr(system.infer_config, "decode_dir", None)
    resolved_scores = None
    if decode_dir:
        matches = list(Path(decode_dir).rglob("scores.json"))
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple scores.json found; set publish.pack_model.decode_dir "
                "to a single decode directory."
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
        publish_cfg=publish_cfg,
        pack_cfg=pack_cfg,
        exp_dir=exp_dir,
        strategy=strategy,
        system=system,
        scores_path=resolved_scores,
    )
    _write_meta(pack_cfg, out_dir)

    logger.info("Packed model to %s", out_dir)
    return out_dir


def _upload_common(
    repo: str,
    src_dir: Path,
    *,
    repo_type: str,
) -> None:
    """Upload artifacts to a Hugging Face repo via huggingface-cli."""
    if shutil.which("huggingface-cli") is None:
        raise RuntimeError("huggingface-cli is required for upload.")

    repo_create_cmd = [
        "huggingface-cli",
        "repo",
        "create",
        repo,
        "--type",
        repo_type,
        "--exist-ok",
    ]
    _run(repo_create_cmd)

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
        system: Object with ``publish_config`` and optionally ``train_config``.
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
    if system.publish_config is None:
        raise RuntimeError("upload_model requires publish_config (publish_config.upload_model).")
    publish_cfg = system.publish_config
    upload_cfg = getattr(publish_cfg, "upload_model", None)
    if upload_cfg is None:
        raise RuntimeError("upload_model requires publish_config.upload_model.")
    repo = getattr(upload_cfg, "hf_repo", None)
    if not repo:
        raise RuntimeError("upload_model requires publish.upload_model.hf_repo")

    exp_dir = Path(system.train_config.exp_dir) if system.train_config else Path.cwd()
    pack_cfg = getattr(publish_cfg, "pack_model", None) or OmegaConf.create({})
    pack_dir = Path(getattr(pack_cfg, "out_dir", exp_dir / "model_pack"))
    if not pack_dir.exists():
        raise RuntimeError(f"Model pack not found: {pack_dir}")

    _upload_common(
        repo,
        pack_dir,
        repo_type="model",
    )
