"""Publication helpers for ESPnet3."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from glob import glob, has_magic
from pathlib import Path
from string import Template
from typing import Any

import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

import espnet2
from espnet3.components.modeling.lightning_module import build_model_summary
from espnet3.utils.logging_utils import get_git_metadata
from espnet3.utils.task_utils import get_espnet_model

logger = logging.getLogger(__name__)

_README_TEMPLATE_BASENAME_ALIASES = {
    "hf_model_repo_readme_template.md": "hf_model_readme.md",
}


def _expand_pack_paths(raw_paths: list[str], recipe_root: Path) -> list[Path]:
    """Expand globbed pack paths and keep unmatched literals as warnings."""
    expanded: list[Path] = []
    seen: set[Path] = set()

    for raw_path in raw_paths:
        path = Path(raw_path)
        if not has_magic(raw_path):
            candidates = [path]
        else:
            pattern = path if path.is_absolute() else recipe_root / path
            candidates = [
                Path(match) for match in sorted(glob(str(pattern), recursive=True))
            ]
            if not candidates:
                logger.warning("Pack path pattern did not match: %s", raw_path)
                continue
        for candidate in candidates:
            normalized = candidate.absolute()
            if normalized in seen:
                continue
            seen.add(normalized)
            expanded.append(candidate)
    return expanded


def _matches_ignore_pattern(relative_path: Path, pattern: str, is_dir: bool) -> bool:
    """Return whether a relative pack path matches an exclude pattern."""
    rel_posix = relative_path.as_posix()
    basename = relative_path.name
    candidates = {rel_posix, basename, f"**/{rel_posix}"}
    if is_dir:
        candidates.update(
            {
                f"{rel_posix}/",
                f"{basename}/",
                f"**/{rel_posix}/",
            }
        )
    return any(fnmatch.fnmatch(candidate, pattern) for candidate in candidates)


def _build_pack_ignore(src_root: Path, excludes: list[str]):
    """Build a copytree ignore callback that matches relative paths."""

    def _ignore(current_dir: str, names: list[str]) -> list[str]:
        ignored: list[str] = []
        current_root = Path(current_dir)
        for name in names:
            candidate = current_root / name
            relative_path = candidate.relative_to(src_root)
            is_dir = candidate.is_dir()
            if any(
                _matches_ignore_pattern(relative_path, pattern, is_dir)
                for pattern in excludes
            ):
                ignored.append(name)
        return ignored

    return _ignore


def _resolve_artifact_paths(
    src_path: str,
    out_dir: Path,
    recipe_root: Path,
) -> tuple[Path, Path]:
    """Validate a named artifact and return its (src, dst) path pair.

    Called by ``pack_model`` before copying each entry in ``pack_cfg.files``
    and ``pack_cfg.yaml_files``. Raises if the source is missing or lives
    outside the recipe root.
    """
    src = Path(src_path)
    if not src.exists():
        raise RuntimeError(f"Artifact does not exist: {src}")
    try:
        dst = out_dir / src.absolute().relative_to(recipe_root)
    except ValueError:
        raise RuntimeError(
            f"Artifact is outside the recipe root and cannot be bundled: {src}. "
            "Consider placing a symlink under the recipe directory instead "
            "(e.g. src/my_tokenizer -> /path/to/tokenizer), so the directory "
            "structure is preserved when copied into the bundle."
        ) from None
    return src, dst


def _copy_path(src: Path, dst: Path, ignore=None) -> None:
    """Copy a file or directory into the bundle.

    Used by ``pack_model`` when copying ``pack_paths`` and named artifact
    files into ``out_dir``. Handles both files and directory trees uniformly.
    """
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _resolve_readme_template_path(readme_template_path: str) -> Path:
    """Resolve a README template path from cwd, repo root, or known aliases."""
    raw_path = Path(readme_template_path)
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = [raw_path]

    if not raw_path.is_absolute():
        candidates.append(repo_root / raw_path)

    for candidate in list(candidates):
        alias_name = _README_TEMPLATE_BASENAME_ALIASES.get(candidate.name)
        if alias_name:
            candidates.append(candidate.with_name(alias_name))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"README template not found: {readme_template_path}")


def _rewrite_paths_for_bundle(
    value: Any,
    recipe_root: Path,
    copied_sources: dict[Path, Path],
    out_dir: Path,
) -> Any:
    """Recursively rewrite source paths in a config to bundle-relative paths.

    Called by ``_write_bundle_config`` for each config written into ``conf/``.
    Walks the config tree and replaces any string value that resolves to a
    path already copied into the bundle with a ``${recipe_dir}/...`` reference,
    so the packed config works portably without absolute paths.

    """
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=False)
    if isinstance(value, dict):
        return {
            k: _rewrite_paths_for_bundle(v, recipe_root, copied_sources, out_dir)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_paths_for_bundle(v, recipe_root, copied_sources, out_dir)
            for v in value
        ]
    if not isinstance(value, str) or not value or "${" in value:
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (recipe_root / candidate).absolute()
    else:
        candidate = candidate.absolute()
    for src_root, dst_root in copied_sources.items():
        if candidate == src_root:
            rel = dst_root.absolute().relative_to(out_dir.resolve()).as_posix()
            return "${recipe_dir}" if rel == "." else f"${{recipe_dir}}/{rel}"
        try:
            suffix = candidate.relative_to(src_root)
        except ValueError:
            continue
        rel = (dst_root / suffix).absolute().relative_to(out_dir.resolve()).as_posix()
        return f"${{recipe_dir}}/{rel}"
    return value


def _write_bundle_config(
    config: DictConfig,
    dest: Path,
    recipe_root: Path,
    copied_sources: dict[Path, Path],
    out_dir: Path,
) -> None:
    """Write a config to the bundle with paths rewritten to bundle-relative form.

    Called by ``pack_model`` for each config (training, inference, metrics,
    publication) to write a portable version into ``conf/``. Paths pointing
    to copied sources are replaced with ``${recipe_dir}/...`` references, and
    ``recipe_dir`` itself is set to ``"."`` so it resolves relative to the
    bundle root at load time.

    """
    plain = OmegaConf.to_container(config, resolve=False)
    rewritten = _rewrite_paths_for_bundle(plain, recipe_root, copied_sources, out_dir)
    cfg = OmegaConf.create(rewritten)
    cfg.recipe_dir = "."
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")


def _resolve_results(
    publication_config: DictConfig | None,
    metrics_config: DictConfig | None,
    inference_config: DictConfig | None,
) -> Path | None:
    """Return metrics.json under inference_dir.

    Called by ``pack_model`` to locate evaluation results before writing the
    README. Checks ``inference_dir / "metrics.json"`` on each provided config
    in priority order (publication → metrics → inference) and returns the
    first existing file.

    """
    for cfg in (publication_config, metrics_config, inference_config):
        inference_dir = getattr(cfg, "inference_dir", None) if cfg else None
        if not inference_dir:
            continue
        metrics_path = Path(inference_dir) / "metrics.json"
        if metrics_path.exists():
            return metrics_path
    return None


def _build_results_table(results_path: Path | None) -> str:
    """Render a markdown results table from metrics.json.

    Called by ``pack_model`` when building the README context. Reads the
    metrics.json produced by the measure stage and formats it as a markdown
    table with test set names as rows and metric names as columns.

    """
    if results_path is None or not results_path.exists():
        return ""
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    # rows[test_name][metric_key] = value
    rows: dict[str, dict[str, str]] = {}
    metric_keys: set[str] = set()
    for metric_name, per_test in results.items():
        if not isinstance(per_test, dict):
            continue
        short_name = str(metric_name).rsplit(".", maxsplit=1)[-1]
        for test_name, value in per_test.items():
            rows.setdefault(str(test_name), {})
            if isinstance(value, dict):
                for k, v in value.items():
                    metric_keys.add(k)
                    rows[str(test_name)][k] = str(v)
            else:
                metric_keys.add(short_name)
                rows[str(test_name)][short_name] = str(value)
    if not rows or not metric_keys:
        return ""
    cols = sorted(metric_keys)
    lines = [
        "## Results",
        "",
        "| dataset | " + " | ".join(cols) + " |",
        "| --- | " + " | ".join("---" for _ in cols) + " |",
    ]
    for test in sorted(rows):
        vals = [rows[test].get(c, "") for c in cols]
        lines.append("| " + " | ".join([test] + vals) + " |")
    lines.append("")
    return "\n".join(lines)


def _infer_system_name(training_config: DictConfig, recipe_root: Path) -> str:
    """Infer a short system name for README rendering."""
    task_value = getattr(training_config, "task", None)
    if isinstance(task_value, str) and task_value:
        parts = task_value.split(".")
        if "systems" in parts:
            systems_index = parts.index("systems")
            if systems_index + 1 < len(parts):
                return parts[systems_index + 1]
        class_name = parts[-1]
        if class_name.endswith("Task") and len(class_name) > 4:
            return class_name[:-4].lower()
        return class_name.lower()
    return recipe_root.name


def _instantiate_publication_model(training_config: DictConfig):
    """Instantiate the training model for README metadata generation."""
    task = training_config.get("task")
    if task:
        model_config = OmegaConf.to_container(training_config.model, resolve=True)
        return get_espnet_model(task, model_config)
    return instantiate(training_config.model)


def _infer_recipe_name(recipe_root: Path) -> str:
    """Return a stable recipe identifier for README metadata."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return recipe_root.relative_to(repo_root).as_posix()
    except ValueError:
        return recipe_root.as_posix()


def _infer_creator() -> str:
    """Return a best-effort creator name for README metadata."""
    return (
        os.environ.get("GIT_AUTHOR_NAME")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or ""
    )


def _describe_pack_strategy(pack_cfg: DictConfig) -> str:
    """Summarize how pack_model builds the publication bundle."""
    parts = ["copy experiment outputs"]
    if getattr(pack_cfg, "include", None):
        parts.append("include extra recipe assets")
    if getattr(pack_cfg, "files", None):
        parts.append("register named artifact files")
    if getattr(pack_cfg, "yaml_files", None):
        parts.append("bundle rewritten YAML configs")
    if getattr(pack_cfg, "exclude", None):
        parts.append("apply exclude filters")
    return "; ".join(parts)


def _build_model_summary_section(model_summary: dict[str, object]) -> str:
    """Render a markdown model summary section for the README."""
    return "\n".join(
        [
            "## Model summary",
            "",
            f"- Class: `{model_summary['class_name']}`",
            f"- Total parameters: `{model_summary['total_params_display']}`",
            (
                "- Learnable parameters: "
                f"`{model_summary['trainable_params_display']}` "
                f"({model_summary['trainable_ratio']:.1f}%)"
            ),
            (
                "- Non-trainable parameters: "
                f"`{model_summary['non_trainable_params_display']}`"
            ),
            f"- Parameter size: `{model_summary['size_display']}`",
            (
                f"- Buffers: `{model_summary['total_buffers_display']}` "
                f"(`{model_summary['buffer_size_display']}`)"
            ),
            (
                f"- Modules: `{model_summary['module_count_display']}` total, "
                f"`{model_summary['leaf_module_count_display']}` leaf"
            ),
            f"- DType composition: `{model_summary['dtype_desc']}`",
            "",
        ]
    )


def _build_model_detail_section(
    model_summary: dict[str, object], include_model_detail: bool
) -> str:
    """Render an optional markdown block with ``repr(model)``."""
    if not include_model_detail:
        return ""
    return "\n".join(
        [
            "## Model structure",
            "",
            "<details><summary>expand</summary>",
            "",
            "```python",
            str(model_summary["repr"]),
            "```",
            "",
            "</details>",
            "",
        ]
    )


def _build_results_note(results_path: Path | None, results_section: str) -> str:
    """Return a README note when metrics are missing or unreadable."""
    if results_section:
        return ""
    if results_path is None:
        return (
            "## Results\n\n"
            "Metrics were not bundled. Run the `measure` stage before "
            "`pack_model` to include evaluation results.\n"
        )
    return (
        "## Results\n\n"
        "A `metrics.json` file was found, but it could not be rendered into "
        "a results table.\n"
    )


def _build_readme_context(
    training_config: DictConfig,
    publication_config: DictConfig,
    out_dir: Path,
    results_path: Path | None,
) -> dict[str, str]:
    """Build default README template values for a publication bundle."""
    pack_cfg = publication_config.pack_model
    recipe_root = Path(training_config.recipe_dir).resolve()
    git_meta = get_git_metadata(recipe_root)
    results_section = _build_results_table(results_path)
    hf_repo = getattr(getattr(publication_config, "upload_model", None), "hf_repo", "")
    usage_load_call = (
        f'model = InferenceModel.from_pretrained("{hf_repo}", trust_user_code=True)'
        if hf_repo
        else (
            "model = InferenceModel.from_packed("
            '"/path/to/packed_model", trust_user_code=True)'
        )
    )
    model_summary_section = ""
    model_detail_section = ""
    try:
        model_summary = build_model_summary(
            _instantiate_publication_model(training_config)
        )
    except Exception as exc:
        logger.warning("README model summary could not be generated: %s", exc)
    else:
        model_summary_section = _build_model_summary_section(model_summary)
        model_detail_section = _build_model_detail_section(
            model_summary,
            include_model_detail=bool(getattr(pack_cfg, "include_model_detail", False)),
        )

    return {
        "corpus": recipe_root.parent.name,
        "creator": _infer_creator(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": (
            f"Packed model bundle generated from `{_infer_recipe_name(recipe_root)}`."
        ),
        "exp_dir": str(getattr(training_config, "exp_dir", "")),
        "git_branch": git_meta.get("branch") or "",
        "git_commit": git_meta.get("commit") or "",
        "git_dirty": git_meta.get("worktree") or "",
        "git_head": git_meta.get("short_commit") or git_meta.get("commit") or "",
        "git_origin": git_meta.get("origin_url") or "",
        "hf_repo": hf_repo,
        "model_detail_section": model_detail_section,
        "model_summary_section": model_summary_section,
        "pack_name": out_dir.name,
        "pack_strategy": _describe_pack_strategy(pack_cfg),
        "recipe": _infer_recipe_name(recipe_root),
        "results_note": _build_results_note(results_path, results_section),
        "results_section": results_section,
        "system": _infer_system_name(training_config, recipe_root),
        "task": _infer_system_name(training_config, recipe_root),
        "task_class": str(getattr(training_config, "task", "") or ""),
        "train_config": OmegaConf.to_yaml(training_config, resolve=True),
        "usage_load_call": usage_load_call,
    }


def _render_readme(template: str, context: dict[str, str]) -> str:
    """Render a README template, dropping lines with empty placeholders.

    Called by ``pack_model`` after resolving the README template path and
    building the context dict. Any line containing a ``${KEY}`` placeholder
    whose value is empty or missing is dropped entirely, so optional sections
    (e.g. results table) don't leave blank lines in the output.

    """
    safe = {k: "" if v is None else str(v) for k, v in context.items()}
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
    lines = []
    for line in template.splitlines():
        keys = pattern.findall(line)
        if keys and any(not safe.get(k, "") for k in keys):
            continue
        lines.append(line)
    return Template("\n".join(lines)).safe_substitute(safe)


def _write_meta(
    out_dir: Path,
    files: dict[str, str],
    yaml_files: dict[str, str],
) -> None:
    """Write meta.yaml into the bundle output directory.

    Called at the end of ``pack_model`` to record all bundled artifact paths
    and environment versions. ``InferenceModel.from_packed`` reads this file
    to locate the inference config.

    """
    meta = {
        "schema_version": 1,
        "files": files,
        "yaml_files": yaml_files,
        "torch": str(torch.__version__),
        "espnet": str(espnet2.__version__),
        "python": sys.version,
    }
    (out_dir / "meta.yaml").write_text(OmegaConf.to_yaml(meta), encoding="utf-8")


def pack_model(
    training_config: DictConfig,
    publication_config: DictConfig,
    inference_config: DictConfig | None = None,
    metrics_config: DictConfig | None = None,
) -> Path:
    """Pack model artifacts for publishing.

    Args:
        training_config: Training configuration with ``recipe_dir`` and
            ``exp_dir``.
        publication_config: Publication configuration with ``pack_model``
            section. ``pack_model.include`` adds extra paths to copy,
            ``pack_model.files`` registers named artifact files, and
            ``pack_model.yaml_files`` registers named YAML configs.
        inference_config: Inference configuration to bundle.
        metrics_config: Metrics configuration to bundle.

    Returns:
        Path to the packed output directory.

    Raises:
        RuntimeError: If artifact paths are invalid.
    """
    pack_cfg = publication_config.pack_model

    recipe_root = Path(training_config.recipe_dir).resolve()

    exp_dir = Path(training_config.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = recipe_root / exp_dir
    exp_dir = exp_dir.resolve()

    # Create (or recreate) the output directory
    out_dir = Path(getattr(pack_cfg, "out_dir", exp_dir / "model_pack"))
    if not out_dir.is_absolute():
        out_dir = recipe_root / out_dir
    out_dir = out_dir.resolve()
    if out_dir.exists():
        if not getattr(pack_cfg, "allow_overwrite", False):
            raise RuntimeError(
                f"out_dir already exists: {out_dir}. "
                "Set pack_model.allow_overwrite: true to overwrite."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    copied_sources: dict[Path, Path] = {}
    files: dict[str, str] = {}
    yaml_files: dict[str, str] = {}

    # Build the ignore filter from config excludes
    excludes = [out_dir.name, "__pycache__"]
    exclude_cfg = getattr(pack_cfg, "exclude", None)
    if exclude_cfg:
        excludes += (
            [str(p) for p in exclude_cfg]
            if isinstance(exclude_cfg, (list, tuple, ListConfig))
            else [str(exclude_cfg)]
        )
    # Copy exp_dir and any extra include paths into the bundle.
    raw_pack_paths: list[str] = [str(exp_dir)]
    include_cfg = getattr(pack_cfg, "include", None)
    if include_cfg:
        raw_pack_paths += (
            [Path(p) for p in include_cfg]
            if isinstance(include_cfg, (list, tuple, ListConfig))
            else [Path(include_cfg)]
        )
    pack_paths = _expand_pack_paths([str(path) for path in raw_pack_paths], recipe_root)

    for path in pack_paths:
        if not path.exists():
            logger.warning("Pack path does not exist: %s", path)
            continue
        try:
            dest = out_dir / path.absolute().relative_to(recipe_root)
        except ValueError:
            logger.warning(
                "Include path is outside recipe root and will be placed at the "
                "bundle root as '%s', losing its original directory structure. "
                "Consider using a symlink under the recipe directory instead: %s",
                path.name,
                path,
            )
            dest = out_dir / path.name
        ignore = _build_pack_ignore(path, excludes) if path.is_dir() else None
        _copy_path(src=path, dst=dest, ignore=ignore)
        copied_sources[path.absolute()] = dest.absolute()

    # Copy named artifact files and register them in meta.
    for key, src_path in dict(getattr(pack_cfg, "files", {}) or {}).items():
        src, dst = _resolve_artifact_paths(src_path, out_dir, recipe_root)
        _copy_path(src=src, dst=dst)
        copied_sources[src.absolute()] = dst.absolute()
        files[key] = dst.relative_to(out_dir).as_posix()

    # Copy named YAML artifacts, rewrite paths, and register them in meta.
    for key, src_path in dict(getattr(pack_cfg, "yaml_files", {}) or {}).items():
        src, dst = _resolve_artifact_paths(src_path, out_dir, recipe_root)
        yaml_config = OmegaConf.load(src)
        _write_bundle_config(yaml_config, dst, recipe_root, copied_sources, out_dir)
        copied_sources[src.absolute()] = dst.absolute()
        yaml_files[key] = dst.relative_to(out_dir).as_posix()

    # Write all configs into conf/ with paths rewritten to bundle-relative form
    conf_dir = out_dir / "conf"
    for meta_key, filename, config in [
        ("training_config", "training.yaml", training_config),
        ("publication_config", "publication.yaml", publication_config),
        ("inference_config", "inference.yaml", inference_config),
        ("metrics_config", "metrics.yaml", metrics_config),
    ]:
        if config is None:
            continue
        dest = conf_dir / filename
        _write_bundle_config(config, dest, recipe_root, copied_sources, out_dir)
        yaml_files[meta_key] = dest.relative_to(out_dir).as_posix()

    # Copy metrics.json if found under inference_dir
    results_path = _resolve_results(
        publication_config, metrics_config, inference_config
    )
    if results_path and results_path.exists():
        shutil.copy2(results_path, out_dir / results_path.name)

    # Render and write README.md from template if configured
    readme_template_path = getattr(pack_cfg, "readme", None)
    if readme_template_path:
        template_path = _resolve_readme_template_path(readme_template_path)
        context = _build_readme_context(
            training_config=training_config,
            publication_config=publication_config,
            out_dir=out_dir,
            results_path=results_path,
        )
        context.update(dict(getattr(pack_cfg, "readme_context", {}) or {}))
        if not context.get("results_section"):
            logger.warning(
                "README will not include a rendered results table. "
                "Run the measure stage before pack_model to bundle metrics.json."
            )
        readme_text = _render_readme(template_path.read_text(encoding="utf-8"), context)
        (out_dir / "README.md").write_text(readme_text, encoding="utf-8")

    _write_meta(out_dir, files=files, yaml_files=yaml_files)
    logger.info("Packed model to %s", out_dir)
    return out_dir


def upload_model(system) -> None:
    """Upload packed model artifacts to a Hugging Face model repo.

    Args:
        system: ESPnet3 system instance with ``publication_config``.

    Raises:
        RuntimeError: If required config values are missing or upload fails.

    Examples:
        >>> upload_model(system)
    """
    publication_cfg = system.publication_config
    upload_cfg = publication_cfg.upload_model
    repo = upload_cfg.hf_repo
    private = bool(getattr(upload_cfg, "private", False))

    # Resolve the pack directory from config
    pack_cfg = getattr(publication_cfg, "pack_model", None) or OmegaConf.create({})
    recipe_root = Path(system.training_config.recipe_dir).resolve()
    exp_dir = Path(system.training_config.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = recipe_root / exp_dir
    exp_dir = exp_dir.resolve()
    pack_dir = Path(getattr(pack_cfg, "out_dir", exp_dir / "model_pack"))
    if not pack_dir.is_absolute():
        pack_dir = recipe_root / pack_dir
    pack_dir = pack_dir.resolve()
    if not pack_dir.exists():
        raise RuntimeError(f"Model pack not found: {pack_dir}")

    hint = (
        "Check `publication_config.upload_model.hf_repo`. "
        "Use the full repo name as `<user-or-org>/<repo>`. "
        "If you meant your own account, replace any other user's namespace "
        "with your Hugging Face username. "
        "If the repo should be private, set `upload_model.private: true`. "
        "Make sure the logged-in token can create and write to that namespace."
    )
    logger.info("Ensuring Hugging Face repo exists: %s", repo)
    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
    except (HfHubHTTPError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to create Hugging Face repo '{repo}': {exc}\n{hint}"
        ) from exc
    logger.info("Uploading %s -> %s", pack_dir, repo)
    try:
        api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(pack_dir))
    except (HfHubHTTPError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to upload model pack to '{repo}': {exc}\n{hint}"
        ) from exc
    logger.info("Upload complete: https://huggingface.co/%s", repo)
