"""Claude Code and Codex command-line agent."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from espnet3.bin.atlas.agents.structured_output import decode_response_object
from espnet3.bin.atlas.core.models import (
    AgentConfig,
    AgentRequest,
    AgentResponse,
    ClaudeConfig,
    CodexConfig,
)
from espnet3.bin.atlas.resources import load_prompt


def _toml_value(value: Any) -> str:
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
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        fields = ", ".join(
            f"{key} = {_toml_value(item)}" for key, item in value.items()
        )
        return "{" + fields + "}"
    raise TypeError(f"Unsupported Codex config value: {type(value).__name__}")


class CliAgent:
    """Invoke the Codex or Claude Code CLI as a subprocess and parse its response.

    Exposes ``run(request, artifact_dir)``. Builds the provider-specific
    command and prompt from an ``AgentRequest``, runs it, parses its
    JSON-lines (Codex) or streaming-JSON (Claude) stdout into an
    ``AgentResponse``, and — on structured output — retries with a
    response-only repair request up to ``config.structured_output_retries``
    times before giving up.
    """

    def __init__(self, config, recipe_dir: Path) -> None:
        """Configure the Codex/Claude Code CLI wrapper.

        Args:
            config: An ``AgentConfig``, or a mapping/OmegaConf node parsed
                via ``AgentConfig.from_config``.
            recipe_dir: Recipe root; used as the default working directory
                when ``config.cwd`` is unset.

        Raises:
            ValueError: If ``structured_output_retries`` is not a
                nonnegative integer.
        """
        if isinstance(config, AgentConfig):
            self.config = config
        else:
            self.config = AgentConfig.from_config(config)
        self.recipe_dir = Path(recipe_dir).resolve()
        self.cwd = Path(self.config.cwd or self.recipe_dir).resolve()
        retries = self.config.structured_output_retries
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("structured_output_retries must be a nonnegative integer")

    def _build_prompt(self, request: AgentRequest) -> str:
        """Render an ``AgentRequest`` into the plain-text prompt sent to Codex/Claude.

        Includes a strict JSON-only response contract section when
        ``request.output_schema`` is non-empty.
        """
        parts = [
            f"Task: {request.task}",
            "",
            "Follow the requested output shape exactly.",
            "",
            "Objective:",
            request.objective.strip(),
            "",
            "Knowledge:",
            request.knowledge.strip(),
            "",
            "Repo Context:",
            request.repo_context.strip(),
            "",
            "Trial History CSV:",
            request.trial_history_csv.strip(),
            "",
            "Latest Metrics:",
            json.dumps(request.latest_metrics, ensure_ascii=False, indent=2),
            "",
            "Additional Context (including log paths when relevant):",
            json.dumps(request.context_files, ensure_ascii=False, indent=2),
            "",
            "Currently Inflight Patches (do NOT propose any of these exactly):",
            json.dumps(request.inflight_patches, ensure_ascii=False, indent=2),
            "",
            "Allowed Actions:",
            ", ".join(request.allowed_actions),
            "",
            "Output JSON Schema:",
            json.dumps(request.output_schema, ensure_ascii=False, indent=2),
        ]
        if request.output_schema:
            parts.insert(2, load_prompt("final_response_contract"))
        return "\n".join(parts)

    def _codex_env(self) -> dict[str, str]:
        """Give Codex a workspace-local home while retaining user credentials."""
        codex_home = self.recipe_dir / ".codex"
        xdg_config_home = codex_home / ".config"
        xdg_cache_home = codex_home / ".cache"
        home_dir = codex_home / ".home"
        for path in (codex_home, xdg_config_home, xdg_cache_home, home_dir):
            path.mkdir(parents=True, exist_ok=True)

        user_codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        for name in ("auth.json", "config.toml", "installation_id", "version.json"):
            source = user_codex_home / name
            destination = codex_home / name
            if source.is_file() and not destination.exists():
                shutil.copyfile(source, destination)
        return {
            "CODEX_HOME": str(codex_home),
            "HOME": str(home_dir),
            "XDG_CONFIG_HOME": str(xdg_config_home),
            "XDG_CACHE_HOME": str(xdg_cache_home),
        }

    def _build_codex_command(
        self, request: AgentRequest, output_path: Path
    ) -> list[str]:
        """Build the ``codex exec`` argv for one request.

        Uses ``codex exec resume <thread_id>`` when ``request.resume_thread_id``
        is set, otherwise a fresh ``codex exec`` in ``self.cwd``; applies
        the configured sandbox/model/config overrides and writes the final
        message to ``output_path``.
        """
        options = self.config.codex
        if request.resume_thread_id:
            command = [
                options.executable,
                "exec",
                "resume",
                request.resume_thread_id,
                "--json",
            ]
        else:
            command = [options.executable, "exec", "--json", "-C", str(self.cwd)]
        if options.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if options.sandbox:
            if request.resume_thread_id:
                if options.sandbox == "danger-full-access":
                    command.append("--dangerously-bypass-approvals-and-sandbox")
                else:
                    command.extend(
                        ["-c", f"sandbox_mode={_toml_value(options.sandbox)}"]
                    )
            else:
                command.extend(["--sandbox", options.sandbox])
        if options.model:
            command.extend(["--model", options.model])
        for key, value in options.config.items():
            command.extend(["-c", f"{key}={_toml_value(value)}"])
        command.extend(options.args)
        command.extend(["--output-last-message", str(output_path)])
        command.append("-")
        return command

    def _build_claude_command(self, request: AgentRequest) -> list[str]:
        """Build the ``claude --print`` argv for one request.

        Applies the configured model/effort/turn-limit/tool options, resumes
        ``request.resume_thread_id`` when set, and disables tools/MCP entirely
        when the request only allows ``"respond_only"``.
        """
        options = self.config.claude
        command = [
            options.executable,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            options.permission_mode,
        ]
        if options.model:
            command.extend(["--model", options.model])
        if options.effort:
            command.extend(["--effort", options.effort])
        if options.max_turns is not None:
            command.extend(["--max-turns", str(options.max_turns)])
        if options.allowed_tools:
            command.extend(["--allowedTools", *options.allowed_tools])
        if options.disallowed_tools:
            command.extend(["--disallowedTools", *options.disallowed_tools])
        command.extend(options.args)
        if request.output_schema:
            command.extend(["--json-schema", json.dumps(request.output_schema)])
        if request.resume_thread_id:
            command.extend(["--resume", request.resume_thread_id])
        if request.allowed_actions == ["respond_only"]:
            command.extend(
                [
                    "--tools",
                    "",
                    "--strict-mcp-config",
                    "--mcp-config",
                    '{"mcpServers":{}}',
                ]
            )
        return command

    @staticmethod
    def _events(stdout: str) -> list[dict[str, Any]]:
        """Parse Codex/Claude's JSON-lines stdout into a list of event dicts, skipping non-JSON lines."""
        events = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    @staticmethod
    def _extract_codex_content(events: list[dict[str, Any]]) -> str:
        """Return the text of the last Codex ``agent_message`` item, or "" if none."""
        for event in reversed(events):
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    return str(item.get("text", "")).strip()
        return ""

    @staticmethod
    def _extract_claude_content(events: list[dict[str, Any]]) -> str:
        """Read the final result, including Claude's native structured output."""
        for event in reversed(events):
            if event.get("type") == "result":
                value = event.get("structured_output")
                if value is None:
                    value = event.get("result", "")
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)
                return str(value).strip() if value is not None else ""
        text = []
        for event in events:
            if event.get("type") != "assistant":
                continue
            message = event.get("message", {})
            for item in message.get("content", []):
                if item.get("type") == "text":
                    text.append(str(item.get("text", "")))
        return "\n".join(text).strip()

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        """Run an agent, asking for bounded response-only repairs of invalid JSON.

        Args:
            request: Task and expected final output fields. Empty output_schema
                permits free-form text and does not trigger JSON repair.
            artifact_dir: Directory for latest artifacts and immutable call history.

        Returns:
            AgentResponse: The valid response or an explicit failure after the
            configured repair limit. Every attempt is retained in the manifest.

        Example:
            >>> # response = agent.run(request, artifact_dir=trial_dir)
            >>> # assert response.status in {"success", "failure"}
        """
        if artifact_dir is None:
            raise ValueError("artifact_dir is required for CliAgent")
        request_dir = (
            Path(artifact_dir) / "agent_requests" / f"request_{uuid.uuid4().hex}"
        )
        request_dir.mkdir(parents=True)
        manifest_path = request_dir / "attempts.json"
        attempts = []
        current = request
        for number in range(self.config.structured_output_retries + 1):
            response = self._run_once(current, artifact_dir)
            attempts.append(
                {
                    "attempt": number + 1,
                    "status": response.status,
                    "message": response.message,
                    "invocation": response.artifacts.get("invocation"),
                }
            )
            manifest_path.write_text(json.dumps(attempts, indent=2), encoding="utf-8")
            response.artifacts["attempts"] = str(manifest_path)
            if (
                response.status == "success"
                or "parse_error" not in response.artifacts
                or number == self.config.structured_output_retries
            ):
                if (
                    response.artifacts.get("parse_error")
                    and number == self.config.structured_output_retries
                ):
                    response.message += (
                        f" (JSON repair limit reached after {number + 1} attempts.)"
                    )
                return response
            instruction = load_prompt("json_repair", error=response.message)
            if not (response.thread_id or current.resume_thread_id):
                instruction += load_prompt(
                    "json_repair_context",
                    objective=request.objective,
                    content=response.content,
                )
            current = replace(
                request,
                objective=instruction,
                knowledge="",
                repo_context="",
                trial_history_csv="",
                latest_metrics={},
                context_files={},
                inflight_patches=[],
                allowed_actions=["respond_only"],
                resume_thread_id=response.thread_id or current.resume_thread_id,
            )
        raise AssertionError("Unreachable structured response retry state")

    def _run_once(self, request: AgentRequest, artifact_dir) -> AgentResponse:
        """Execute one CLI invocation and validate its final structured response."""
        if artifact_dir is None:
            raise ValueError("artifact_dir is required for CliAgent")
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        invocation_dir = artifact_dir / "agent_calls" / f"call_{uuid.uuid4().hex}"
        invocation_dir.mkdir(parents=True)
        # Retain legacy flat artifacts before their first overwrite after this
        # upgrade; subsequent calls already have their own immutable copies.
        if len(list(invocation_dir.parent.iterdir())) == 1:
            previous = invocation_dir.parent / "legacy"
            for name in (
                "agent_prompt.txt",
                "agent_output.txt",
                "agent_stdout.log",
                "agent_stderr.log",
                "agent_events.jsonl",
                "agent_parse_error.json",
            ):
                source = artifact_dir / name
                if source.is_file():
                    previous.mkdir(exist_ok=True)
                    shutil.copy2(source, previous / name)
        (artifact_dir / "agent_parse_error.json").unlink(missing_ok=True)

        def _write_artifact(path: Path, value: str) -> None:
            """Keep both the latest artifact and an immutable per-invocation copy."""
            path.write_text(value, encoding="utf-8")
            (invocation_dir / path.name).write_text(value, encoding="utf-8")

        prompt = self._build_prompt(request)
        prompt_path = artifact_dir / "agent_prompt.txt"
        output_path = artifact_dir / "agent_output.txt"
        stdout_path = artifact_dir / "agent_stdout.log"
        stderr_path = artifact_dir / "agent_stderr.log"
        events_path = artifact_dir / "agent_events.jsonl"
        _write_artifact(prompt_path, prompt)
        for path in (output_path, stdout_path, stderr_path, events_path):
            _write_artifact(path, "")
        artifacts = {
            "prompt": str(prompt_path),
            "output": str(output_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "events": str(events_path),
            "invocation": str(invocation_dir),
        }

        if self.config.provider == "codex":
            command = self._build_codex_command(request, output_path)
            extra_env = self._codex_env()
        else:
            command = self._build_claude_command(request)
            extra_env = {}
        try:
            proc = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=self.cwd,
                env={**os.environ, **extra_env, **self.config.env},
                timeout=self.config.timeout_sec,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                for path, value in (
                    (stdout_path, exc.stdout),
                    (stderr_path, exc.stderr),
                    (events_path, exc.stdout),
                ):
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    _write_artifact(path, value or "")
            message = (
                f"Agent executable was not found: {command[0]}"
                if isinstance(exc, FileNotFoundError)
                else f"Agent CLI timed out after {self.config.timeout_sec} seconds."
            )
            return AgentResponse(
                status="failure", message=message, content="", artifacts=artifacts
            )

        _write_artifact(stdout_path, proc.stdout)
        _write_artifact(stderr_path, proc.stderr)
        _write_artifact(events_path, proc.stdout)
        events = self._events(proc.stdout)
        if self.config.provider == "codex":
            self._write_codex_usage(artifact_dir, request, command, events)
        thread_id = next(
            (
                str(event[key])
                for event in events
                for key in ("thread_id", "session_id")
                if event.get(key)
            ),
            None,
        )
        if proc.returncode != 0:
            return AgentResponse(
                status="failure",
                message=(
                    f"Agent CLI failed with exit code {proc.returncode}: "
                    f"{' '.join(shlex.quote(part) for part in command)}"
                ),
                content=proc.stderr.strip() or proc.stdout.strip(),
                thread_id=thread_id,
                artifacts=artifacts,
            )

        if self.config.provider == "codex":
            content = output_path.read_text(encoding="utf-8").strip()
            if not content:
                content = self._extract_codex_content(events)
        else:
            content = self._extract_claude_content(events)
        _write_artifact(output_path, content)
        if self.config.provider == "claude":
            result_event = next(
                (event for event in reversed(events) if event.get("type") == "result"),
                None,
            )
            if result_event is None or result_event.get("is_error"):
                return AgentResponse(
                    status="failure",
                    message=(
                        "Claude stream has no final result event."
                        if result_event is None
                        else f"Claude reported an error result: {result_event.get('subtype')}"
                    ),
                    content=content,
                    thread_id=thread_id,
                    artifacts=artifacts,
                )
        try:
            structured = decode_response_object(content)
            self._validate_output(structured, request.output_schema)
        except ValueError as exc:
            if request.output_schema:
                error_path = artifact_dir / "agent_parse_error.json"
                _write_artifact(
                    error_path,
                    json.dumps({"task": request.task, "error": str(exc)}, indent=2),
                )
                artifacts["parse_error"] = str(error_path)
                return AgentResponse(
                    status="failure",
                    message=f"Invalid structured agent output: {exc}",
                    content=content,
                    thread_id=thread_id,
                    artifacts=artifacts,
                )
            structured = {}
        return AgentResponse(
            status="success",
            message="Agent CLI completed.",
            content=content,
            structured=structured,
            thread_id=thread_id,
            artifacts=artifacts,
        )

    @staticmethod
    def _validate_output(value: dict, schema: dict) -> None:
        """Raise ``ValueError`` when ``value`` does not satisfy the JSON Schema."""
        if not schema:
            return
        from jsonschema import Draft202012Validator

        errors = sorted(
            Draft202012Validator(schema).iter_errors(value), key=lambda e: list(e.path)
        )
        if errors:
            error = errors[0]
            where = ".".join(str(part) for part in error.path) or "root"
            raise ValueError(f"{where}: {error.message}")

    @staticmethod
    def _parse_structured_output(content: str) -> dict[str, Any]:
        """Parse complete JSON output, returning an empty mapping for invalid replies.

        Args:
            content: Agent response text.

        Returns:
            dict[str, Any]: Parsed JSON object, or an empty mapping on invalid output.

        Example:
            >>> CliAgent._parse_structured_output('{"status": "ok"}')
            {'status': 'ok'}
        """
        try:
            return decode_response_object(content)
        except ValueError:
            return {}

    @staticmethod
    def _find_usage_objects(value) -> list[dict[str, Any]]:
        """Recursively collect token-usage dicts from a nested Codex event structure.

        Finds any ``"usage"`` sub-dict and any dict directly carrying token
        count fields (``input_tokens``, ``output_tokens``, etc.), at any
        nesting depth within ``value``.
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
                found.extend(CliAgent._find_usage_objects(nested))
        elif isinstance(value, list):
            for item in value:
                found.extend(CliAgent._find_usage_objects(item))
        return found

    @staticmethod
    def _write_codex_usage(
        artifact_dir: Path,
        request: AgentRequest,
        command: list[str],
        events: list[dict[str, Any]],
    ) -> None:
        """Write a Codex call's token-usage summary to ``agent_usage.json`` and append to ``agent_usage_history.jsonl``."""
        usage_records = [
            usage
            for event in events
            for usage in CliAgent._find_usage_objects(event)
            if usage
        ]
        summary = {
            "task": request.task,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "command": command,
            "usage_records": usage_records,
            "last_usage": usage_records[-1] if usage_records else {},
            "event_count": len(events),
        }
        (artifact_dir / "agent_usage.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (artifact_dir / "agent_usage_history.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
