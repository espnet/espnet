"""Codex (``codex exec``) provider."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from espnet3.bin.atlas.agents.providers import (
    Invocation,
    ParsedOutput,
    Provider,
    find_thread_id,
    parse_json_events,
)
from espnet3.bin.atlas.core.models import AgentRequest


def _format_toml_value(value: Any) -> str:
    """Serialize a value accepted by Codex's ``-c key=value`` option."""
    if value is None:
        raise ValueError("Codex config overrides cannot contain null values")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        fields = ", ".join(
            f"{key} = {_format_toml_value(item)}" for key, item in value.items()
        )
        return "{" + fields + "}"
    raise TypeError(f"Unsupported Codex config value: {type(value).__name__}")


def _collect_usage_objects(value: Any) -> list[dict[str, Any]]:
    """Recursively collect token-usage dicts from a nested Codex event.

    Finds any ``"usage"`` sub-dict and any dict directly carrying token count
    fields (``input_tokens``, ``output_tokens``, ...), at any nesting depth.
    """
    found = []
    if isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict):
            found.append(usage)
        token_keys = {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "reasoning_output_tokens",
        }
        if any(key in value for key in token_keys):
            found.append(
                {
                    key: value[key]
                    for key in token_keys
                    if isinstance(value.get(key), (int, float))
                }
            )
        for nested in value.values():
            found.extend(_collect_usage_objects(nested))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_usage_objects(item))
    return found


class CodexProvider(Provider):
    """Call ``codex exec`` and read its JSON-lines output."""

    def build_command(
        self,
        request: AgentRequest,
        prompt: str,
        *,
        cwd: Path,
        output_path: Path,
        schema_path: Path,
    ) -> Invocation:
        """Build the ``codex exec`` argv for one request.

        Uses ``codex exec resume <thread_id>`` when ``request.resume_thread_id``
        is set, otherwise a fresh ``codex exec`` in ``cwd``; applies the
        configured sandbox/model/config overrides and writes the final message
        to ``output_path``. A request's schema is passed as ``--output-schema``,
        so the final message is constrained to it. The prompt goes to stdin.
        """
        options = self.config.codex
        command = [options.executable, "exec"]
        if request.resume_thread_id:
            command.extend(["resume", request.resume_thread_id])
        else:
            command.extend(["-C", str(cwd)])
        command.append("--json")
        if options.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if options.sandbox:
            if request.resume_thread_id:
                if options.sandbox == "danger-full-access":
                    command.append("--dangerously-bypass-approvals-and-sandbox")
                else:
                    command.extend(
                        ["-c", f"sandbox_mode={_format_toml_value(options.sandbox)}"]
                    )
            else:
                command.extend(["--sandbox", options.sandbox])
        if options.model:
            command.extend(["--model", options.model])
        for key, value in OmegaConf.to_container(options.config).items():
            command.extend(["-c", f"{key}={_format_toml_value(value)}"])
        command.extend(str(arg) for arg in options.args)
        if request.output_schema:
            command.extend(["--output-schema", str(schema_path)])
        command.extend(["--output-last-message", str(output_path)])
        command.append("-")
        return Invocation(command, stdin=prompt)

    def build_env(self, recipe_dir: Path) -> dict[str, str]:
        """Give Codex a workspace-local home while retaining user credentials."""
        codex_home = recipe_dir / ".codex"
        paths = {
            "CODEX_HOME": codex_home,
            "HOME": codex_home / ".home",
            "XDG_CONFIG_HOME": codex_home / ".config",
            "XDG_CACHE_HOME": codex_home / ".cache",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        user_codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        for name in ("auth.json", "config.toml", "installation_id", "version.json"):
            source = user_codex_home / name
            destination = codex_home / name
            if source.is_file() and not destination.exists():
                shutil.copyfile(source, destination)
        return {name: str(path) for name, path in paths.items()}

    def parse(self, stdout: str, output_path: Path) -> ParsedOutput:
        """Read the final message file, else the last ``agent_message`` event."""
        events = parse_json_events(stdout)
        content = ""
        if output_path.is_file():
            content = output_path.read_text(encoding="utf-8").strip()
        if not content:
            for event in reversed(events):
                item = event.get("item", {})
                if event.get("type") == "item.completed" and (
                    item.get("type") == "agent_message"
                ):
                    content = str(item.get("text", "")).strip()
                    break
        return ParsedOutput(content, find_thread_id(events))

    def record_usage(
        self,
        artifact_dir: Path,
        request: AgentRequest,
        argv: list[str],
        stdout: str,
    ) -> None:
        """Write the call's token usage to ``agent_usage.json`` and its history."""
        events = parse_json_events(stdout)
        usage_records = [
            usage
            for event in events
            for usage in _collect_usage_objects(event)
            if usage
        ]
        summary = {
            "task": request.task,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "command": argv,
            "usage_records": usage_records,
            "last_usage": usage_records[-1] if usage_records else {},
            "event_count": len(events),
        }
        (artifact_dir / "agent_usage.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        history = artifact_dir / "agent_usage_history.jsonl"
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
