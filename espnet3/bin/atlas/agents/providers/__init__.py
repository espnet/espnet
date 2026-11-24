"""Coding-agent CLI providers: how to call one CLI and how to read its reply.

``CliAgent`` runs the subprocess and keeps the artifacts; a provider only knows
one CLI's command line, environment and output format. A provider must make
its CLI constrain the final reply to the request's JSON Schema (Atlas does not
parse free-form text), so a CLI without such an option cannot be added.
Add a provider by subclassing :class:`Provider` and listing it in
:data:`PROVIDERS` and :func:`create_provider`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from espnet3.bin.atlas.core.models import AgentRequest

PROVIDERS = ("codex", "claude")


@dataclass
class Invocation:
    """One subprocess call: its argument list and what to write to its stdin.

    Args:
        argv: Program and arguments.
        stdin: Text for the process's standard input, or ``None`` for none.
    """

    argv: list[str]
    stdin: str | None = None


@dataclass
class ParsedOutput:
    """What a provider read from a finished call's standard output.

    Args:
        content: The agent's final reply text.
        thread_id: Id to resume the conversation with, or ``None``.
        error: Why the CLI itself reported failure although it exited
            normally, or ``None``.
    """

    content: str
    thread_id: str | None = None
    error: str | None = None


def parse_json_events(stdout: str) -> list[dict[str, Any]]:
    """Parse JSON-lines standard output into event dicts, skipping other lines.

    Args:
        stdout: Standard output of an agent CLI that prints one JSON object per
            line.

    Returns:
        list[dict[str, Any]]: The JSON objects, in order.

    Example:
        >>> parse_json_events('{"type": "a"}\\nnot json\\n')
        [{'type': 'a'}]
    """
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def find_thread_id(events: list[dict[str, Any]]) -> str | None:
    """Return the first ``thread_id`` or ``session_id`` found in ``events``.

    Args:
        events: Parsed JSON-lines events.

    Returns:
        str | None: The conversation id, or ``None`` if no event carries one.

    Example:
        >>> find_thread_id([{"type": "x"}, {"session_id": "s1"}])
        's1'
    """
    return next(
        (
            str(event[key])
            for event in events
            for key in ("thread_id", "session_id")
            if event.get(key)
        ),
        None,
    )


class Provider:
    """Interface of one coding-agent CLI.

    Args:
        config: The complete ``atlas.agent`` settings (see
            ``resources/agent_defaults.yaml``).
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def build_command(
        self,
        request: AgentRequest,
        prompt: str,
        *,
        cwd: Path,
        output_path: Path,
        schema_path: Path,
    ) -> Invocation:
        """Build the subprocess call for one request.

        Args:
            request: The task to run.
            prompt: The rendered prompt text.
            cwd: Working directory the process runs in.
            output_path: File a CLI may write its final message to.
            schema_path: File holding ``request.output_schema`` as JSON (written
                only when the request has a schema).

        Returns:
            Invocation: Arguments and stdin text.
        """
        raise NotImplementedError

    def build_env(self, recipe_dir: Path) -> dict[str, str]:
        """Return extra environment variables for the process.

        Args:
            recipe_dir: Recipe root.

        Returns:
            dict[str, str]: Variables added to the inherited environment.
        """
        return {}

    def parse(self, stdout: str, output_path: Path) -> ParsedOutput:
        """Read the final reply from a finished call.

        Args:
            stdout: The process's standard output.
            output_path: The file passed to :meth:`build_command`.

        Returns:
            ParsedOutput: Reply text, conversation id and any CLI-reported error.
        """
        raise NotImplementedError

    def record_usage(
        self,
        artifact_dir: Path,
        request: AgentRequest,
        argv: list[str],
        stdout: str,
    ) -> None:
        """Write token-usage records for a finished call, if the CLI reports any.

        Args:
            artifact_dir: Directory for the call's artifacts.
            request: The task that ran.
            argv: The command that ran.
            stdout: The process's standard output.
        """


def create_provider(config: DictConfig) -> Provider:
    """Create the provider named by ``config.provider``.

    Args:
        config: The complete ``atlas.agent`` settings.

    Returns:
        Provider: The matching provider.

    Raises:
        ValueError: If the name is not in :data:`PROVIDERS`.

    Example:
        >>> create_provider(OmegaConf.create({"provider": "codex"}))  # doctest: +SKIP
        <...CodexProvider object at ...>
    """
    from espnet3.bin.atlas.agents.providers.claude import ClaudeProvider
    from espnet3.bin.atlas.agents.providers.codex import CodexProvider

    classes = {"codex": CodexProvider, "claude": ClaudeProvider}
    if config.provider not in classes:
        raise ValueError(f"Unknown agent provider: {config.provider!r}")
    return classes[config.provider](config)
