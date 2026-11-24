"""Claude Code (``claude --print``) provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from espnet3.bin.atlas.agents.providers import (
    Invocation,
    ParsedOutput,
    Provider,
    find_thread_id,
    parse_json_events,
)
from espnet3.bin.atlas.core.models import AgentRequest


class ClaudeProvider(Provider):
    """Call ``claude --print`` and read its streaming-JSON output."""

    def build_command(
        self,
        request: AgentRequest,
        prompt: str,
        *,
        cwd: Path,
        output_path: Path,
        schema_path: Path,
    ) -> Invocation:
        """Build the ``claude --print`` argv for one request.

        Applies the configured model/effort/turn-limit/tool options, passes the
        output JSON Schema natively, resumes ``request.resume_thread_id`` when
        set, and disables tools/MCP entirely when the request only allows
        ``"respond_only"``. The prompt goes to stdin.
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
            command.extend(["--allowedTools", *map(str, options.allowed_tools)])
        if options.disallowed_tools:
            command.extend(["--disallowedTools", *map(str, options.disallowed_tools)])
        command.extend(str(arg) for arg in options.args)
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
        return Invocation(command, stdin=prompt)

    def parse(self, stdout: str, output_path: Path) -> ParsedOutput:
        """Read the final ``result`` event, including native structured output."""
        events = parse_json_events(stdout)
        result = next(
            (event for event in reversed(events) if event.get("type") == "result"),
            None,
        )
        error = None
        if result is None:
            error = "Claude stream has no final result event."
        elif result.get("is_error"):
            error = f"Claude reported an error result: {result.get('subtype')}"
        return ParsedOutput(_extract_content(events), find_thread_id(events), error)


def _extract_content(events: list[dict[str, Any]]) -> str:
    """Return the final result text, or the assistant's text if there is none."""
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
        for item in event.get("message", {}).get("content", []):
            if item.get("type") == "text":
                text.append(str(item.get("text", "")))
    return "\n".join(text).strip()
