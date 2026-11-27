"""Parse <file path="...">content</file> blocks from agent output and write them."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from espnet3.bin.atlas.core.models import AgentRequest, StageResult
from espnet3.bin.atlas.core.utils import get_utc_now, save_yaml
from espnet3.bin.atlas.stages.base import BaseStage

_FILE_BLOCK_RE = re.compile(
    r'<file\s+path="(?P<path>[^"]+)">\n?(?P<content>.*?)\n?</file>',
    re.DOTALL,
)


def parse_file_blocks(text: str) -> list[tuple[str, str]]:
    """Extract every ``<file path="...">content</file>`` block from agent output.

    Args:
        text: Agent response text, possibly containing one or more
            ``<file>`` blocks interleaved with other prose.

    Returns:
        list[tuple[str, str]]: ``(relative_path, content)`` pairs, in the
        order they appear in ``text``.

    Example:
        >>> parse_file_blocks('<file path="src/a.py">x = 1</file>')
        [('src/a.py', 'x = 1')]
    """
    return [
        (m.group("path"), m.group("content")) for m in _FILE_BLOCK_RE.finditer(text)
    ]


def _is_allowed(rel_path: str, allowed_dirs: list[str]) -> bool:
    """Whether ``rel_path`` is inside (or equal to) one of ``allowed_dirs``."""
    p = Path(rel_path)
    return any(p == Path(d) or Path(d) in p.parents for d in allowed_dirs)


def _read_files(recipe_dir: Path, allowed_dirs: list[str]) -> str:
    """Render every file under ``allowed_dirs`` as ``<file path="...">`` blocks."""
    parts: list[str] = []
    for rel in allowed_dirs:
        d = recipe_dir / rel
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                rel_path = p.relative_to(recipe_dir)
                try:
                    content = p.read_text(encoding="utf-8")
                except Exception:
                    content = "(binary or unreadable)"
                parts.append(f'<file path="{rel_path}">\n{content}\n</file>')
    return "\n\n".join(parts)


def _read_reference_files(paths: list[str]) -> str:
    """Render each reference file's content as a labeled markdown code block."""
    parts: list[str] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            parts.append(f"# {raw}\n(file not found)")
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as e:
            content = f"(unreadable: {e})"
        parts.append(f"# {p.name}\n```python\n{content}\n```")
    return "\n\n".join(parts)


def _fetch_url(url: str) -> str:
    """Fetch URL via urllib and strip HTML tags to plain text."""
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"(failed to fetch {url}: {exc})"
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or f"(empty response from {url})"


def _fetch_doc_urls(urls: list[str], cache_dir: Path) -> str:
    """Fetch each URL and return concatenated plain-text. Results are cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for url in urls:
        safe_name = re.sub(r"[^\w.-]", "_", url.lstrip("https://"))[:80] + ".txt"
        cache_path = cache_dir / safe_name
        if not cache_path.exists():
            content = _fetch_url(url)
            cache_path.write_text(content, encoding="utf-8")
        else:
            content = cache_path.read_text(encoding="utf-8")
        parts.append(f"## Doc: {url}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def _read_training_config(trial_dir: Path) -> str:
    """Return the trial's resolved training config text, or a not-found placeholder."""
    cfg_path = trial_dir / "resolved_training_config.yaml"
    if not cfg_path.exists():
        return "(resolved_training_config.yaml not found)"
    return cfg_path.read_text(encoding="utf-8")


def _find_latest_test_failure(trial_dir: Path) -> str | None:
    """Return the latest ``run_tests`` error log, or ``None`` if none failed."""
    test_runs_dir = trial_dir / "test_runs"
    if not test_runs_dir.exists():
        return None
    for run_dir in reversed(sorted(test_runs_dir.iterdir())):
        error_log = run_dir / "error.log"
        if error_log.exists():
            return error_log.read_text(encoding="utf-8")
    return None


def _create_next_iteration_dir(trial_dir: Path) -> Path:
    """Create and return the next numbered ``write_files_iterations/iter_NNN`` dir."""
    base = trial_dir / "write_files_iterations"
    base.mkdir(parents=True, exist_ok=True)
    n = len(list(base.iterdir())) + 1
    d = base / f"iter_{n:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_prompt(
    trial,
    recipe_dir: Path,
    allowed_dirs: list[str],
    reference_files_text: str,
    doc_text: str,
    training_config_text: str,
    prev_error: str | None,
) -> str:
    """Build the write_files agent prompt from the trial's rationale and context.

    Assembles the prior rationale, any previous test failure (for a fix
    attempt), the resolved training config, reference files, fetched
    documentation, the current allowed-directory contents, and output-format
    instructions into one prompt string.
    """
    current_files = _read_files(recipe_dir, allowed_dirs)
    allowed_str = "\n".join(f"  - {d}" for d in allowed_dirs)

    sections: list[str] = [
        "## Task: write_files",
        "",
        "You previously proposed the following architecture change:",
        "Rationale: "
        f"{trial.rationale.splitlines()[0] if trial.rationale else '(none)'}",
        "",
    ]

    if prev_error:
        sections += [
            "## Previous implementation FAILED unit tests — fix these errors",
            "",
            "```",
            prev_error.strip(),
            "```",
            "",
        ]

    if training_config_text:
        sections += [
            "## Resolved training config (model dimensions, vocab_size, etc.)",
            "",
            "```yaml",
            training_config_text.strip(),
            "```",
            "",
        ]

    if reference_files_text:
        sections += [
            "## Reference files (model interface / input format examples)",
            "",
            reference_files_text,
            "",
        ]

    if doc_text:
        sections += [
            "## ESPnet3 documentation "
            "(how to wire custom models into the training system)",
            "",
            doc_text,
            "",
        ]

    sections += [
        "## Current source files in allowed directories",
        "",
        current_files or "(no files yet)",
        "",
        "## Instructions",
        "",
        "Output BOTH implementation files AND unit tests using "
        '<file path="..."> blocks.',
        f"Allowed directories (relative to recipe root):\n{allowed_str}",
        "",
        "Format each file as:",
        "",
        '    <file path="src/model.py">',
        "    # complete file contents — not a diff",
        "    </file>",
        "",
        "Requirements for unit tests:",
        "- Use actual torch.Tensor inputs (e.g. torch.randn(...)) — no mocks.",
        "- Use model dimensions from the training config above "
        "(vocab_size, d_model, etc.).",
        "- Test forward pass with realistic batch shapes.",
        "- Assert output shapes and dtypes explicitly.",
        "- Tests must run with plain `pytest` — no external fixtures or data files.",
        "",
        "ESPnet3 wiring rules:",
        "- Set `model._target_: src.model.MyModel` in the config patch "
        "(not the task field).",
        "- The model's __init__ args must match the config keys exactly.",
        "- Keep `task` field unset "
        "(ESPnet3 instantiates the model directly via Hydra).",
        "",
        "General rules:",
        "- Output COMPLETE file contents, not diffs.",
        "- Only write files inside the allowed directories.",
        "- After all <file> blocks, write a one-line summary.",
    ]
    if prev_error:
        sections.append(
            "- This is a fix attempt — address EVERY error in the failure log above."
        )

    return "\n".join(sections)


class WriteFilesStage(BaseStage):
    """Ask the agent to generate implementation + unit tests via <file path=""> blocks.

    Reads ``code_edit.reference_files`` (local paths) and fetches
    ``code_edit.doc_urls`` (web pages) to include in the prompt alongside the
    resolved training config, so the agent has full context on model dimensions
    and ESPnet3 wiring conventions.

    On re-runs after a failed ``run_tests``, the previous pytest error log is
    prepended to the prompt so the agent can fix the issues.
    """

    study_lock_name = "study_controller"

    def run(self, context) -> StageResult:
        """Ask the agent to write implementation + test files, then apply them.

        Builds the prompt (training config, reference files, fetched docs,
        current allowed-dir contents, and any previous test failure), calls
        the agent, parses ``<file>`` blocks from its response, backs up and
        overwrites each allowed file, and records what was written/skipped.

        Args:
            context: The node run's ``StageContext``; must have a current
                trial.

        Returns:
            StageResult: ``status="success"`` if ``code_edit.allowed_dirs``
            is unset (no-op), the agent produced no ``<file>`` blocks, or
            files were written successfully; ``status="failure"`` if the
            agent call itself failed.

        Example:
            >>> result = WriteFilesStage().run(stage_context)
            >>> result.status
            'success'
        """
        trial = context.current_trial
        assert trial is not None

        code_edit_cfg = getattr(context.config.atlas, "code_edit", None)
        allowed_dirs: list[str] = list(getattr(code_edit_cfg, "allowed_dirs", []) or [])
        reference_files: list[str] = list(
            getattr(code_edit_cfg, "reference_files", []) or []
        )
        doc_urls: list[str] = list(getattr(code_edit_cfg, "doc_urls", []) or [])

        if not allowed_dirs:
            return StageResult(
                status="success",
                message="code_edit.allowed_dirs not configured — skipping write_files",
                payload={"trial_id": trial.trial_id},
            )

        trial_dir = context.get_trial_dir()
        iter_dir = _create_next_iteration_dir(trial_dir)
        doc_cache_dir = context.study_dir / ".doc_cache"

        prev_error = _find_latest_test_failure(trial_dir)
        reference_files_text = (
            _read_reference_files(reference_files) if reference_files else ""
        )
        doc_text = _fetch_doc_urls(doc_urls, doc_cache_dir) if doc_urls else ""
        training_config_text = _read_training_config(trial_dir)

        prompt = _build_prompt(
            trial,
            context.recipe_dir,
            allowed_dirs,
            reference_files_text,
            doc_text,
            training_config_text,
            prev_error,
        )
        (iter_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        knowledge_path = context.study_dir / "knowledge" / "knowledge_pack.md"
        knowledge = (
            knowledge_path.read_text(encoding="utf-8")
            if knowledge_path.exists()
            else ""
        )

        request = AgentRequest(
            task="write_files",
            objective=(context.study_dir / "program.md").read_text(encoding="utf-8"),
            knowledge=knowledge,
            repo_context=prompt,
            trial_history_csv="",
            latest_metrics={},
            context_files={},
            allowed_actions=["file_write"],
            output_schema={},
            resume_thread_id=trial.codex_thread_id or None,
        )

        response = context.agent.run(request, artifact_dir=iter_dir)

        if response.status != "success":
            return StageResult(
                status="failure",
                message=f"write_files agent failed: {response.message}",
                payload={"trial_id": trial.trial_id},
            )

        raw_output = response.content or ""
        (iter_dir / "agent_output.txt").write_text(raw_output, encoding="utf-8")

        file_blocks = parse_file_blocks(raw_output)
        if not file_blocks:
            return StageResult(
                status="success",
                message="Agent produced no <file> blocks — no files written",
                payload={"trial_id": trial.trial_id},
            )

        written: list[str] = []
        skipped: list[str] = []
        backup_dir = iter_dir / "backup"

        for rel_path, content in file_blocks:
            if not _is_allowed(rel_path, allowed_dirs):
                skipped.append(rel_path)
                continue
            full_path = context.recipe_dir / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if full_path.exists():
                backup_path = backup_dir / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(full_path, backup_path)
            full_path.write_text(content, encoding="utf-8")
            written.append(rel_path)

        save_yaml(
            iter_dir / "result.yaml",
            {
                "trial_id": trial.trial_id,
                "written": written,
                "skipped_not_allowed": skipped,
                "is_fix_attempt": prev_error is not None,
                "doc_urls_fetched": doc_urls,
                "reference_files_used": reference_files,
                "timestamp": get_utc_now(),
            },
        )

        if skipped:
            context.logger.warning(
                "write_files: skipped %s (not in allowed_dirs)", skipped
            )

        return StageResult(
            status="success",
            message=f"Wrote {len(written)} file(s): {written}",
            payload={"trial_id": trial.trial_id, "written": written},
            artifacts={"iter_dir": str(iter_dir)},
        )
