"""Agent interface."""

from __future__ import annotations

from typing import Protocol


from espnet3.bin.atlas.core.models import AgentRequest, AgentResponse


class AgentClient(Protocol):
    """Interface implemented by every coding-agent client Atlas can call.

    Satisfied by :class:`espnet3.bin.atlas.agents.cli.CliAgent` (real
    Codex/Claude Code subprocess calls) and by test doubles that return a
    fixed :class:`AgentResponse` without spawning a process.
    """

    def run(self, request: AgentRequest, artifact_dir=None) -> AgentResponse:
        """Run one agent task and return its parsed response.

        Args:
            request: The task, context, and allowed actions for the agent.
            artifact_dir: Directory the agent may write output artifacts
                under, or ``None`` to use an implementation default.

        Returns:
            AgentResponse: The agent's status, message, and structured
            output.

        Example:
            >>> response = agent.run(request)
            >>> response.status
            'success'
        """
