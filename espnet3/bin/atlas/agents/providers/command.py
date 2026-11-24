"""Provider for any CLI that takes a prompt and prints its reply."""

from __future__ import annotations

from pathlib import Path

from espnet3.bin.atlas.agents.providers import Invocation, ParsedOutput, Provider
from espnet3.bin.atlas.core.models import AgentRequest

PLACEHOLDER = "{prompt}"


class CommandProvider(Provider):
    """Run the command given in ``atlas.agent.command`` and read its standard output.

    ``command.argv`` is the program and its arguments. With
    ``command.prompt_via: stdin`` (the default) the prompt is written to the
    process's standard input; with ``arg`` the text ``{prompt}`` inside one of
    the arguments is replaced by the prompt. Whatever the process prints is the
    reply: there is no session resume, usage record or native schema option, so
    structured replies rely on the prompt, validation and the JSON repair retry.

    Example:
        ``provider: command`` with ``command: {argv: [my-agent, -p, "{prompt}"],
        prompt_via: arg}``.
    """

    def build_command(
        self, request: AgentRequest, prompt: str, *, cwd: Path, output_path: Path
    ) -> Invocation:
        """Build the call from ``command.argv`` and ``command.prompt_via``.

        Raises:
            ValueError: If ``argv`` is empty, ``prompt_via`` is unknown, or
                ``prompt_via: arg`` has no ``{prompt}`` in any argument.
        """
        options = self.config.command
        argv = [str(arg) for arg in options.argv]
        if not argv:
            raise ValueError("atlas.agent.command.argv must name the program to run")
        via = str(options.prompt_via)
        if via == "stdin":
            return Invocation(argv, stdin=prompt)
        if via != "arg":
            raise ValueError("atlas.agent.command.prompt_via must be 'stdin' or 'arg'")
        if not any(PLACEHOLDER in arg for arg in argv):
            raise ValueError(
                "atlas.agent.command.argv needs a {prompt} argument for prompt_via: arg"
            )
        return Invocation([arg.replace(PLACEHOLDER, prompt) for arg in argv])

    def parse(self, stdout: str, output_path: Path) -> ParsedOutput:
        """Treat the whole standard output as the reply."""
        return ParsedOutput(stdout.strip())
